for i in {0..1}
do
    echo Running state $i
    mkdir -p DATA/trained/state${i}
    mkdir -p DATA/Images_${i}/
    mkdir -p DATA/Parameters_${i}/
    python rbm_train.py --model ising  --batch 100 --hidden 8 --epoch 1000   --k 20  --imgout ./DATA/Images_${i}/ --txtout ./DATA/Parameters_${i}/ --ising_size 8 --train ../magneto/state_new${i}.txt
    mv trained_rbm.* DATA/trained/state${i}
done
