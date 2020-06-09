##################
# training scripts
##################
#Run this to start the container;
docker run --gpus all --ipc=host --rm -it -v $PWD:/workspace --network=host icaruszyz/large-scale-training:dialogpt bash
#Then within the container, run this
python demo.py --model small --data custom --train train_ehealth_agent.tsv --valid valid_ehealth_agent.tsv


##################
# interact scripts
##################
python interact_auto.py --model_name_or_path ./models/small/ --agent_checkpoint ./models/output_model/GPT2.1e-05.32.1gpu.2020-05-22173929/GP2-pretrain-step-25000.pkl --customer_checkpoint ./models/output_model/GPT2.1e-05.32.1gpu.2020-05-23152925/GP2-pretrain-step-20000.pkl --generation_length 100 --seed 1234
python interact_auto.py --model_name_or_path ./models/small/ --agent_checkpoint ./models/output_model/GPT2.1e-05.32.1gpu.2020-05-22173929/GP2-pretrain-step-25000.pkl --customer_checkpoint ./models/output_model/GPT2.1e-05.32.1gpu.2020-05-23152925/GP2-pretrain-step-20000.pkl --generation_length 100 --seed 1235
python interact_auto.py --model_name_or_path ./models/small/ --agent_checkpoint ./models/output_model/GPT2.1e-05.32.1gpu.2020-05-22173929/GP2-pretrain-step-25000.pkl --customer_checkpoint ./models/output_model/GPT2.1e-05.32.1gpu.2020-05-23152925/GP2-pretrain-step-20000.pkl --generation_length 100 --seed 1236
python interact_auto.py --model_name_or_path ./models/small/ --agent_checkpoint ./models/output_model/GPT2.1e-05.32.1gpu.2020-05-22173929/GP2-pretrain-step-25000.pkl --customer_checkpoint ./models/output_model/GPT2.1e-05.32.1gpu.2020-05-23152925/GP2-pretrain-step-20000.pkl --generation_length 100 --seed 1237
python interact_auto.py --model_name_or_path ./models/small/ --agent_checkpoint ./models/output_model/GPT2.1e-05.32.1gpu.2020-05-22173929/GP2-pretrain-step-25000.pkl --customer_checkpoint ./models/output_model/GPT2.1e-05.32.1gpu.2020-05-23152925/GP2-pretrain-step-20000.pkl --generation_length 100 --seed 1238
# predict customer utterance
#python interact.py --model_name_or_path ./models/small/ --load_checkpoint ./models/output_model/GPT2.1e-05.32.1gpu.2020-05-23152925/GP2-pretrain-step-20000.pkl --generation_length 50
#python interact.py --model_name_or_path ./models/medium/ --load_checkpoint ./models/output_model/GPT2.1e-05.6.1gpu.2020-05-25151237/GP2-pretrain-step-140000.pkl --generation_length 50

# predict agent utterance
#python interact.py --model_name_or_path ./models/small/ --load_checkpoint ./models/output_model/GPT2.1e-05.32.1gpu.2020-05-22173929/GP2-pretrain-step-25000.pkl --generation_length 50
#python interact.py --model_name_or_path ./models/medium/ --load_checkpoint ./models/output_model/GPT2.1e-05.6.1gpu.2020-05-28062702/GP2-pretrain-step-130000.pkl --generation_length 50
python interact.py --model_name_or_path ./models/small/ --load_checkpoint ./models/output_model/GPT2.1e-05.32.1gpu.2020-05-22173929/GP2-pretrain-step-25000.pkl --generation_length 50 --seed 12345 --use_gpu --number_of_examples 75 --top_p 0.95
