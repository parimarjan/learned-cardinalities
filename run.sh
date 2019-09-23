#python3 main.py --db_name dmv --num_samples_per_template 1000 --template_dir templates/dmv3/ --algs ourpgm --losses qerr --pgm_alg_name chow-liu --test 0

#python3 main.py --db_name dmv --num_samples_per_template 1000 --template_dir templates/dmv3/ --algs ourpgm --losses qerr --pgm_alg_name chow-liu --test 0 --use_svd 1 --num_singular_vals 5

#python3 main.py --db_name dmv --num_samples_per_template 1000 --template_dir \
#templates/dmv3/ --algs ourpgm --losses qerr --pgm_alg_name chow-liu --test 0 \
#--use_svd 0 --num_singular_vals 5 --cl_recompute 1

#python3 main.py --db_name dmv --num_samples_per_template 1000 --template_dir \
#templates/dmv3/ --algs ourpgm --losses qerr --pgm_alg_name chow-liu --test 0 \
#--use_svd 1 --num_singular_vals 5 --cl_recompute 1

python3 main.py --db_name higgs --num_samples_per_template 1000 --template_dir \
templates/higgs/ --algs ourpgm --losses qerr --pgm_alg_name chow-liu --test 0

python3 main.py --db_name higgs --num_samples_per_template 1000 --template_dir \
templates/higgs/ --algs ourpgm --losses qerr --pgm_alg_name chow-liu --test 0 \
--use_svd 1 --num_singular_vals 5 \

#python3 main.py --db_name higgs --num_samples_per_template 1000 --template_dir \
#templates/higgs/ --algs ourpgm --losses qerr --pgm_alg_name chow-liu --test 0 \
#--use_svd 1 --num_singular_vals 5 --cl_recompute 1 \

#python3 main.py --db_name higgs --num_samples_per_template 1000 --template_dir \
#templates/higgs/ --algs ourpgm --losses qerr --pgm_alg_name chow-liu --test 0 \
#--use_svd 1 --num_singular_vals 5 --cl_recompute 0

#python3 main.py --db_name osm2 --num_samples_per_template 10000 --template_dir \
#templates/osm/ --algs ourpgm --losses qerr --pgm_alg_name chow-liu --test 0

#python3 main.py --db_name osm2 --num_samples_per_template 10000 --template_dir \
#templates/osm/ --algs ourpgm --losses qerr --pgm_alg_name chow-liu --test 0 \
#--use_svd 1 --num_singular_vals 5 \

#python3 main.py --db_name osm2 --num_samples_per_template 10000 --template_dir \
#templates/osm/ --algs ourpgm --losses qerr --pgm_alg_name chow-liu --test 0 \
#--use_svd 1 --num_singular_vals 5 --cl_recompute 1 \

#python3 main.py --db_name osm2 --num_samples_per_template 10000 --template_dir \
#templates/osm/ --algs ourpgm --losses qerr --pgm_alg_name chow-liu --test 0 \
#--use_svd 1 --num_singular_vals 5 --cl_recompute 0
