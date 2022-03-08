# Script for running a series of active learning tests with RWLL

#python al_test_rwll.py 
#python send_email.py --message "Done with mnist-evenodd test"
#python al_test_rwll.py --dataset mnist
python test_al_beta.py --dataset mstar
python accuracy_al_beta.py --dataset mstar
python send_email.py --message "Done with mstar test"
python test_al_beta.py --dataset mstar-evenodd
python accuracy_al_beta.py --dataset mstar-evenodd
python send_email.py --message "Done with mstar-evenodd test"
#python al_test_rwll.py --dataset fashionmnist
#python send_email.py --message "Done with fashionmnist test"
#python al_test_rwll.py --dataset fashionmnist-evenodd
#python send_email.py --message "Done with fashionmnist-evenodd test"
#python al_test_rwll.py --dataset cifar --metric aet
#python send_email.py --message "Done with cifar test"
