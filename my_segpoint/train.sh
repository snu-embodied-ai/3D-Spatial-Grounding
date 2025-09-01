export CUDA_VISIBLE_DEVICES=2,3,4

accelerate launch \
    --config_file cfg/launch/launch.yaml \
    --dynamo_backend no \
    run.py \
    --trainer_cfg cfg/trainer.yaml