USE_IMAGENET_PRETRAINED = True # otherwise use detectron, but that doesnt seem to work?!?

# Change these to match where your annotations and images are
VCR_IMAGES_DIR = '/mnt/home/yangshao/vcr/vcr1/vcr1images'
# VCR_IMAGES_DIR = '/mnt/gs18/scratch/users/yangshao/vcr/vcr1/vcr1images'
# VCR_ANNOTS_DIR = '/mnt/home/yangshao/vcr/new_data'
VCR_DIR = '/mnt/home/yangshao/vcr/r2c'
# VCR_DIR = '/mnt/gs18/scratch/users/yangshao/vcr/r2c'

double_flag = False

gumble_temperature = 1.0
gumble_decay = 0.0001
vae_inference_sample_ct = 50
kl_weight = 1.0

