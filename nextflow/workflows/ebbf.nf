/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    IMPORT MODULES / SUBWORKFLOWS / FUNCTIONS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/
include { EBBF_PREPROCESS           } from '../modules/local/preprocess'
include { EBBF_TRAIN                } from '../modules/local/train'
include { EBBF_INFER                } from '../modules/local/infer'
include { EBBF_POSTPROCESS          } from '../modules/local/postprocess'
include { EBBF_REPORTING            } from '../modules/local/reporting'
include { MULTIQC                   } from '../modules/nf-core/multiqc/main'
include { paramsSummaryMap          } from 'plugin/nf-schema'
include { paramsSummaryMultiqc      } from '../subworkflows/nf-core/utils_nfcore_pipeline'
include { softwareVersionsToYAML    } from '../subworkflows/nf-core/utils_nfcore_pipeline'
include { methodsDescriptionText    } from '../subworkflows/local/utils_nfcore_ebbf_pipeline'

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
    ch_multiqc_files = channel.empty()

    //
    // STEP 1: Preprocess - Convert zarr to TMA format
    // If preprocess_dir is provided, skip preprocessing and use existing files
    //
    if (params.preprocess_dir) {
        // Use existing preprocessing outputs
        ch_dataset = ch_samplesheet
            .map { meta, zarr_dir -> [ meta, file("${params.preprocess_dir}/stdata.npz") ] }
        ch_transcripts = ch_samplesheet
            .map { meta, zarr_dir -> [ meta, file("${params.preprocess_dir}/transcripts.parquet") ] }
    } else {
        EBBF_PREPROCESS(ch_samplesheet)
        ch_versions = ch_versions.mix(EBBF_PREPROCESS.out.versions.first())
        ch_dataset = EBBF_PREPROCESS.out.dataset
        ch_transcripts = EBBF_PREPROCESS.out.transcripts
    }

    //
    // STEP 2: Train or skip
    // model_weights + resume_training  -> train with --resume_from
    // model_weights + !resume_training -> skip training, use weights directly
    // !model_weights                   -> train from scratch
    //
    if (params.model_weights && !params.resume_training) {
        // Skip training, use provided weights directly for inference
        ch_weights = ch_dataset
            .map { meta, dataset -> [ meta, file(params.model_weights) ] }
    } else {
        // Train the model (optionally resuming from model_weights)
        EBBF_TRAIN(ch_dataset)
        ch_versions = ch_versions.mix(EBBF_TRAIN.out.versions.first())
        ch_weights = EBBF_TRAIN.out.weights
    }

    //
    // STEP 3: Infer - Run inference with the trained model
    // Transcripts are derived from zarr_dir: points/transcripts/
    // Generate worker channels for parallel inference
    //
    ch_infer_input = ch_weights
        .join(ch_dataset)
        .join(ch_samplesheet)
        .combine(Channel.of(0..<params.n_infer_workers))
        .map { meta, weights, dataset, zarr_dir, worker_id ->
            [ meta, weights, dataset, zarr_dir, params.n_infer_workers, worker_id ]
        }

    EBBF_INFER(ch_infer_input)
    ch_versions = ch_versions.mix(EBBF_INFER.out.versions.first())

    //
    // STEP 4: Postprocess - Aggregate inference results
    // Wait for all inference workers to complete before postprocessing
    //
    ch_postprocess_input = EBBF_INFER.out.results
        .groupTuple()
        .map { meta, parquet_list ->
            [ meta, parquet_list.flatten() ]
        }
        .join(ch_dataset)
        .join(ch_transcripts)
        .join(ch_samplesheet)

    EBBF_POSTPROCESS(ch_postprocess_input)
    ch_versions = ch_versions.mix(EBBF_POSTPROCESS.out.versions.first())

    //
    // STEP 5: Reporting - Generate HTML comparison reports
    // Can be disabled with --reporting false to save compute time
    //
    if (params.reporting) {
        ch_reporting_input = EBBF_POSTPROCESS.out.anndata
            .join(EBBF_POSTPROCESS.out.results)
            .join(ch_transcripts)
            .join(ch_samplesheet)

        EBBF_REPORTING(ch_reporting_input)
        ch_versions = ch_versions.mix(EBBF_REPORTING.out.versions.first())
    }

    //
    // Collate and save software versions
    //
    def topic_versions = Channel.topic("versions")
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
            name: 'nf_core_'  +  'ebbf_software_'  + 'mqc_'  + 'versions.yml',
            sort: true,
            newLine: true
        ).set { ch_collated_versions }


    //
    // MODULE: MultiQC
    //
    ch_multiqc_config        = channel.fromPath(
        "$projectDir/assets/multiqc_config.yml", checkIfExists: true)
    ch_multiqc_custom_config = params.multiqc_config ?
        channel.fromPath(params.multiqc_config, checkIfExists: true) :
        channel.empty()
    ch_multiqc_logo          = params.multiqc_logo ?
        channel.fromPath(params.multiqc_logo, checkIfExists: true) :
        channel.empty()

    summary_params      = paramsSummaryMap(
        workflow, parameters_schema: "nextflow_schema.json")
    ch_workflow_summary = channel.value(paramsSummaryMultiqc(summary_params))
    ch_multiqc_files = ch_multiqc_files.mix(
        ch_workflow_summary.collectFile(name: 'workflow_summary_mqc.yaml'))
    ch_multiqc_custom_methods_description = params.multiqc_methods_description ?
        file(params.multiqc_methods_description, checkIfExists: true) :
        file("$projectDir/assets/methods_description_template.yml", checkIfExists: true)
    ch_methods_description                = channel.value(
        methodsDescriptionText(ch_multiqc_custom_methods_description))

    ch_multiqc_files = ch_multiqc_files.mix(ch_collated_versions)
    ch_multiqc_files = ch_multiqc_files.mix(
        ch_methods_description.collectFile(
            name: 'methods_description_mqc.yaml',
            sort: true
        )
    )

    MULTIQC (
        ch_multiqc_files.collect(),
        ch_multiqc_config.toList(),
        ch_multiqc_custom_config.toList(),
        ch_multiqc_logo.toList(),
        [],
        []
    )

    emit:
    postprocess_results = EBBF_POSTPROCESS.out.results
    reports             = params.reporting ? EBBF_REPORTING.out.reports : channel.empty()
    multiqc_report      = MULTIQC.out.report.toList()
    versions            = ch_versions

}

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    THE END
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/
