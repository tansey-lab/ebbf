

python ./scripts/preprocess.py

python ./src/ebbf/train_parallel.py --config "./configs/example.yaml"

## Can run multiple jobs in parallel here (worker_id should span 0..n_workers-1)
python ./src/ebbf/inference/infer.py --model_directory "./results/coad/revived-dawn-299" --transcripts_fname "./data/spatch/coad/spatialdata.zarr/points/transcripts/points.parquet" --n_workers 1 --worker_id 0

python ./src/ebbf/inference/post_process.py --model_directory "./results/coad/revived-dawn-299" --tma_directory "/data1/tanseyw/projects/tosh/nuc_diffusion/data/spatch/coad"
