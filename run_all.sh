# Script for running a series of active learning tests with Beta Learning

# python test_al_beta.py --dataset mstar
# python accuracy_al_beta.py --dataset mstar
# python send_email.py --message "Done with mstar test"
# python test_al_beta.py --dataset mstar-evenodd
# python accuracy_al_beta.py --dataset mstar-evenodd
# python send_email.py --message "Done with mstar-evenodd test"
# python test_al_beta.py --dataset mnist-evenodd
# python accuracy_al_beta.py --dataset mnist-evenodd
# python send_email.py --message "Done with mnist test"
# python test_al_beta.py --dataset mnist
# python accuracy_al_beta.py --dataset mnist
# python send_email.py --message "Done with mnist-evenodd test"

python test_al_beta.py --dataset fashionmnist
python accuracy_al_beta.py --dataset fashionmnist
python send_email.py --message "Done with fashionmnist test"
python test_al_beta.py --dataset fashionmnist-evenodd
python accuracy_al_beta.py --dataset fashionmnist-evenodd
python send_email.py --message "Done with fashionmnist-evenodd test"
