# python src/ogbn-proteins/teacher_train.py --lbd_pred 0.1 --lbd_embd 0.01 --hidden 64 --layer 28 --train_bn 40 --test_bn 5
python src/ogbn-proteins/student_train.py --lbd_pred 0.1 --lbd_embd 0.01 --hidden 64 --layer 28 --train_bn 40 --test_bn 5
