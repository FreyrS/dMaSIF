# Load environment
python -W ignore -u main_training.py --experiment_name dMaSIF_site_1layer_15A --batch_size 64 --embedding_layer dMaSIF --site True --single_protein True --random_rotation True --radius 15.0 --n_layers 1
python -W ignore -u main_training.py --experiment_name dMaSIF_site_1layer_5A --batch_size 64 --embedding_layer dMaSIF --site True --single_protein True --random_rotation True --radius 5.0 --n_layers 1
python -W ignore -u main_training.py --experiment_name dMaSIF_site_1layer_9A --batch_size 64 --embedding_layer dMaSIF --site True --single_protein True --random_rotation True --radius 9.0 --n_layers 1

python -W ignore -u main_training.py --experiment_name dMaSIF_site_3layer_15A --batch_size 64 --embedding_layer dMaSIF --site True --single_protein True --random_rotation True --radius 15.0 --n_layers 3
python -W ignore -u main_training.py --experiment_name dMaSIF_site_3layer_5A --batch_size 64 --embedding_layer dMaSIF --site True --single_protein True --random_rotation True --radius 5.0 --n_layers 3
python -W ignore -u main_training.py --experiment_name dMaSIF_site_3layer_9A --batch_size 64 --embedding_layer dMaSIF --site True --single_protein True --random_rotation True --radius 9.0 --n_layers 3
