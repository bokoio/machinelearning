#git config --global user.email "feboscato@gmail.com"
#git config --global user.name "Felipe Nedeff"
#ssh-keygen -t rsa -b 4096 -C "feboscato@gmail.com"
#ssh-add ~/.ssh/id_rsa
#/home/pippo/.ssh/id_rsa.pub

#Initialize the local directory as a Git repository.
#git init
git add .
git commit -m $1
#git remote add origin git@github.com:bokoio/machinelearning.git
#git remote remove origin
#git remote -v
git push origin master
