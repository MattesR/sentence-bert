# Repository for comparision of embedding-based and term-based similarity Search

As of now, elasticsearch is used for term-based similarity search. Documents(mails in this first case) are store as Documents, containing inner documents for text units.  
faiss is used as an index in order to store the embeddings of text units. 


## Datasets
[Enron Dataset](https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz)

## Installed packages  
There's a list of installed packages in the `setup` folder, in order to make it easier to install all necessary packages for the project on a new machine.  
The installed package list is an error-wreckingball. If something throws an error, you might not want
to find out after years of stackoverflowing around, that you need to in fact install swig or whatever.
Therefore I recommend (to myself) to install all these packages before trying out the project.
This is the output of the following command:
`comm -23 <(apt-mark showmanual | sort -u) <(gzip -dc /var/log/installer/initial-status.gz | sed -n 's/^Package: //p' | sort -u)`

I got it from this post: [here](https://askubuntu.com/questions/2389/generating-list-of-manually-installed-packages-and-querying-individual-packages)


There are some packages (like screen, xclip, make, curl) which you don't need for the project. This is more like a 
'get me started on a new machine type' of list. Some of them are probably already installed on a new machine as well.  
Feel free to omit from installing packages from this list.  

`xargs sudo apt-get install < ./setup/package_list.txt`  
or   
`xargs sudo apt-get -y install < ./setup/package_list.txt`  
will install all packages from `./setup/package_list.txt`

