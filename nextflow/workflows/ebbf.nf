/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    IMPORT MODULES / SUBWORKFLOWS / FUNCTIONS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/
include { EBBF_PREPROCESS           } from '../modules/local/preprocess'
include { EBBF_TRAIN                } from '../modules/local/train'
include { EBBF_INFER                } from '../modules/local/infer'
include { EBBF_POSTPROCESS_PREPARE  } from '../modules/local/postprocess'
include { EBBF_POSTPROCESS_MAP      } from '../modules/local/postprocess'
include { EBBF_POSTPROCESS_MERGE    } from '../modules/local/postprocess'
include { EBBF_BOUNDARIES           } from '../modules/local/boundaries'
include { EBBF_REPORTING            } from '../modules/local/reporting'
include { paramsSummaryMap          } from 'plugin/nf-schema'
include { softwareVersionsToYAML    } from '../subworkflows/nf-core/utils_nfcore_pipeline'

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    HELPERS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

// Samplesheet columns that select which stages a sample skips. They arrive as
// meta entries (see assets/schema_input.json) and are removed from meta before
// any process sees it.
def OVERRIDE_KEYS = ['preprocess_dir', 'model_weights', 'inference_results']

// file() with a glob returns a List for many matches, a bare Path for one, and
// an empty List for none — normalise all three to a List.
def globShards(String dir) {
    def matched = file("${dir}/part-*.parquet")
    def as_list = matched instanceof List ? matched : (matched ? [matched] : [])
    return as_list.flatten().findAll { p -> p }
}

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    RUN MAIN WORKFLOW
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

workflow EBBF {

    take:
    ch_samplesheet // channel: [ meta, zarr_dir ]

    main:

    ch_versions = channel.empty()

    //
    // STEP 0: Resolve per-sample stage overrides
    //
    // `preprocess_dir`, `model_weights` and `inference_results` may be given
    // per-sample as samplesheet columns, each falling back to the like-named
    // global param when its cell is empty. Deciding per-sample rather than
    // globally is what lets a single run mix samples that sit at different
    // stages — e.g. re-entering a long run where most samples already have
    // inference shards on disk and one still needs to finish training.
    //
    // The override keys are stripped from `meta` right away. `meta` is folded
    // into every task hash, so leaving them there would make editing one
    // sample's row invalidate every *other* sample's cached tasks on -resume.
    //
    ch_input = ch_samplesheet
        .map { meta, zarr_dir ->
            def opts = [
                preprocess_dir   : meta.preprocess_dir    ?: params.preprocess_dir,
                model_weights    : meta.model_weights     ?: params.model_weights,
                inference_results: meta.inference_results ?: params.inference_results,
            ]
            [ meta.findAll { k, _v -> !(k in OVERRIDE_KEYS) }, zarr_dir, opts ]
        }

    ch_sample_zarr = ch_input.map { meta, zarr_dir, _opts -> [ meta, zarr_dir ] }

    //
    // STEP 1: Preprocess - Convert zarr to TMA format
    // Samples carrying a preprocess_dir reuse its stdata.npz / transcripts.parquet.
    //
    def tissue_boundaries_file = params.tissue_boundaries ? file(params.tissue_boundaries, checkIfExists: true) : []

    ch_preprocess_branch = ch_input.branch { _meta, _zarr_dir, opts ->
        existing: opts.preprocess_dir as boolean
        to_run  : true
    }

    EBBF_PREPROCESS(
        ch_preprocess_branch.to_run.map { meta, zarr_dir, _opts -> [ meta, zarr_dir, tissue_boundaries_file ] }
    )
    ch_versions = ch_versions.mix(EBBF_PREPROCESS.out.versions.first())

    ch_existing_preprocess = ch_preprocess_branch.existing
        .map { meta, _zarr_dir, opts ->
            def base = file(opts.preprocess_dir, checkIfExists: true)
            [
                meta,
                file("${base}/stdata.npz", checkIfExists: true),
                file("${base}/transcripts.parquet", checkIfExists: true),
            ]
        }

    ch_dataset = EBBF_PREPROCESS.out.dataset
        .mix(ch_existing_preprocess.map { meta, dataset, _tx -> [ meta, dataset ] })
    ch_transcripts = EBBF_PREPROCESS.out.transcripts
        .mix(ch_existing_preprocess.map { meta, _dataset, tx -> [ meta, tx ] })

    //
    // STEP 2–3: Train + infer, skipped per-sample when inference_results already
    // supplies that sample's part-*.parquet shards.
    //
    ch_infer_branch = ch_input.branch { _meta, _zarr_dir, opts ->
        existing: opts.inference_results as boolean
        to_run  : true
    }

    ch_existing_shards = ch_infer_branch.existing
        .map { meta, _zarr_dir, opts ->
            def base = file(opts.inference_results, checkIfExists: true)
            if (!base.isDirectory()) {
                error("inference_results must be a directory: ${base}")
            }
            // Accept either a per-sample subdirectory or a directory of shards
            // pointed at directly by that sample's row.
            def parts = globShards("${base}/${meta.id}")
            if (parts.isEmpty()) {
                parts = globShards("${base}")
            }
            if (parts.isEmpty()) {
                error("No part-*.parquet inference shards found for sample ${meta.id}. Tried: ${base}/${meta.id}/part-*.parquet and ${base}/part-*.parquet")
            }
            parts.sort { a, b -> a.name <=> b.name }
            [ meta, parts ]
        }
        .join(ch_dataset)

    //
    // STEP 2: Train or skip
    // model_weights + resume_training  -> train with --resume_from
    // model_weights + !resume_training -> skip training, use weights directly
    // !model_weights                   -> train from scratch (EBBF_TRAIN still
    //                                     picks up any checkpoint left in
    //                                     ${outdir}/checkpoints/<id> via its
    //                                     beforeScript, so an interrupted
    //                                     sample continues from its last epoch)
    //
    ch_train_branch = ch_infer_branch.to_run
        .map { meta, _zarr_dir, opts -> [ meta, opts ] }
        .join(ch_dataset)
        .branch { _meta, opts, _dataset ->
            skip_train: opts.model_weights && !params.resume_training
            to_train  : true
        }

    EBBF_TRAIN(ch_train_branch.to_train.map { meta, _opts, dataset -> [ meta, dataset ] })
    ch_versions = ch_versions.mix(EBBF_TRAIN.out.versions.first())

    ch_weights = EBBF_TRAIN.out.weights
        .mix(
            ch_train_branch.skip_train.map { meta, opts, _dataset ->
                [ meta, file(opts.model_weights, checkIfExists: true) ]
            }
        )

    //
    // STEP 3: Infer - Run inference with the trained model
    // Transcripts are derived from zarr_dir: points/transcripts/
    // Generate worker channels for parallel inference
    //
    ch_infer_input = ch_weights
        .join(ch_dataset)
        .join(ch_sample_zarr)
        .combine(channel.of(0..<params.n_infer_workers))
        .map { meta, weights, dataset, zarr_dir, worker_id ->
            [ meta, weights, dataset, zarr_dir, params.n_infer_workers, worker_id ]
        }

    EBBF_INFER(ch_infer_input)
    ch_versions = ch_versions.mix(EBBF_INFER.out.versions.first())

    //
    // STEP 4 (input): Aggregate inference worker outputs per sample
    //
    // Each sample emits exactly n_infer_workers tuples into `results` (one per
    // worker task). Passing `size` lets groupTuple release each sample's group
    // as soon as its own workers finish, instead of waiting for the whole
    // EBBF_INFER channel to drain — otherwise no sample can advance to
    // EBBF_POSTPROCESS_PREPARE until ALL samples finish inference.
    ch_new_shards = EBBF_INFER.out.results
        .groupTuple(size: params.n_infer_workers)
        .map { meta, parquet_list ->
            [ meta, parquet_list.flatten() ]
        }
        .join(ch_dataset)

    ch_postprocess_prepare_input = ch_existing_shards.mix(ch_new_shards)

    //
    // STEP 4: Postprocess prepare — from inference outputs or from inference_results shards
    //
    EBBF_POSTPROCESS_PREPARE(ch_postprocess_prepare_input)
    ch_versions = ch_versions.mix(EBBF_POSTPROCESS_PREPARE.out.versions.first())

    ch_postprocess_map_input = EBBF_POSTPROCESS_PREPARE.out.step4
        .join(ch_dataset)
        .combine(channel.of(0..<params.n_postprocess_workers))
        .map { meta, step4_parquet, dataset_file, worker_id ->
            [ meta, step4_parquet, dataset_file, params.n_postprocess_workers, worker_id ]
        }

    EBBF_POSTPROCESS_MAP(ch_postprocess_map_input)
    ch_versions = ch_versions.mix(EBBF_POSTPROCESS_MAP.out.versions.first())

    // Same pattern: each sample emits n_postprocess_workers tuples into `mapped`,
    // so give groupTuple the size to release each sample independently rather than
    // barriering EBBF_POSTPROCESS_MERGE behind every sample's MAP step.
    ch_postprocess_merge_input = EBBF_POSTPROCESS_MAP.out.mapped
        .groupTuple(size: params.n_postprocess_workers)
        .map { meta, mapped_parts ->
            [ meta, mapped_parts.flatten() ]
        }
        .join(ch_dataset)
        .join(ch_transcripts)
        .join(ch_sample_zarr)
        .join(EBBF_POSTPROCESS_PREPARE.out.noise_meta)

    EBBF_POSTPROCESS_MERGE(ch_postprocess_merge_input)
    ch_versions = ch_versions.mix(EBBF_POSTPROCESS_MERGE.out.versions.first())

    //
    // STEP 5a: Build first-class cell_boundaries_ebbf.parquet output
    //
    ch_boundaries_input = EBBF_POSTPROCESS_MERGE.out.results
        .join(ch_transcripts)

    EBBF_BOUNDARIES(ch_boundaries_input)
    ch_versions = ch_versions.mix(EBBF_BOUNDARIES.out.versions.first())

    //
    // STEP 5b: Reporting - HTML/PDF analysis & plots from h5ad + EBBF boundaries
    //   (default Xenium boundaries are loaded inside reporting from the zarr dir
    //   for comparison plots; we don't bake them into the boundaries parquet)
    // Can be disabled with --reporting false to save compute time
    //
    if (params.reporting) {
        ch_reporting_input = EBBF_POSTPROCESS_MERGE.out.anndata
            .join(EBBF_POSTPROCESS_MERGE.out.anndata_unfiltered)
            .join(EBBF_BOUNDARIES.out.boundaries)
            .join(ch_sample_zarr)
            .join(ch_dataset)

        EBBF_REPORTING(ch_reporting_input)
        ch_versions = ch_versions.mix(EBBF_REPORTING.out.versions.first())
    }

    //
    // Collate and save software versions
    //
    def topic_versions = channel.topic("versions")
        .distinct()
        .branch { entry ->
            versions_file: entry instanceof Path
            versions_tuple: true
        }

    def topic_versions_string = topic_versions.versions_tuple
        .map { process, tool, version ->
            [ process[process.lastIndexOf(':')+1..-1], "  ${tool}: ${version}" ]
        }
        .groupTuple(by:0)
        .map { process, tool_versions ->
            tool_versions.unique().sort()
            "${process}:\n${tool_versions.join('\n')}"
        }

    softwareVersionsToYAML(ch_versions.mix(topic_versions.versions_file))
        .mix(topic_versions_string)
        .collectFile(
            storeDir: "${params.outdir}/pipeline_info",
            name: 'nf_core_ebbf_software_versions.yml',
            sort: true,
            newLine: true
        )

    emit:
    postprocess_results = EBBF_POSTPROCESS_MERGE.out.results
    reports             = params.reporting ? EBBF_REPORTING.out.reports : channel.empty()
    versions            = ch_versions

}

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    THE END
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/
