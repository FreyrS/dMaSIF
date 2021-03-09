# Load environment
python -W ignore -u main_training.py --experiment_name DGCNN_site_1layer_k200 --batch_size 64 --embedding_layer DGCNN --site True --single_protein True --device cuda:0 --n_layers 1 --random_rotation True --k 200
python -W ignore -u main_training.py --experiment_name DGCNN_site_1layer_k100 --batch_size 64 --embedding_layer DGCNN --site True --single_protein True --device cuda:0 --n_layers 1 --random_rotation True --k 100

python -W ignore -u main_training.py --experiment_name DGCNN_site_3layer_k200 --batch_size 64 --embedding_layer DGCNN --site True --single_protein True --device cuda:0 --n_layers 3 --random_rotation True --k 200
python -W ignore -u main_training.py --experiment_name DGCNN_site_3layer_k100 --batch_size 64 --embedding_layer DGCNN --site True --single_protein True --device cuda:0 --n_layers 3 --random_rotation True --k 100