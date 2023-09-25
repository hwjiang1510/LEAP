CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port 8888 --nproc_per_node=1 \
demo.py --cfg ./config/demo/demo_224_real.yaml
# ./config/demo/demo_224_amazon.yaml    
# ./config/demo/demo_224_real.yaml