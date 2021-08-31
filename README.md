# CloudAssignment_M20AIE244
CSL7510 Assessment 1: Virtual Machines &amp; Dockers

# To explore different datasets and classifier in Machine learning.
 
## Process
Below are the steps followed-
1) Created a machine learning code using Python and Streamlit which includes the pre-defined libraries(Streamlit,Matplotlib,Scikit Learn).
I have written code using Visual Studio code. So to import the libraries, I have create one virtual environment in VS code.
Below are the commands for creating a virtual environemnt.
      - python -m venv --system-site-packages .\tf  (Create a virtual environment and tf is the name of my virtual environment)
      - .\tf\Scripts\activate (Activate the virtual environment)
      - pip install --upgrade pip (Upgrade to the latest version of pip)
      - pip list (Check pre installed packages in virtual environment)
      - pip insatall matplotlib 
      - pip install scikit-learn
      -pip install streamlit
2) Created a docker file in the VS code.
3) Created a requirement text file in the VS code.
4) Install Docker in the local system and using Powershell run below commands
     - Create a Docker image using below command, name of my docker image is testimage
         - docker build -t testimage .

     - After docker image create a container,name of my docker container is kavitacontainer
         - docker run --name kavitacontainer -p 8501:8501 testimage
5) Because web app is created using streatmlit, so we can run container now and app will directly run in the browser 
   using the link
   - localhost:8051



## App Functionality
This machine learning app will explore different datasets and classifier . 
For an example in the left side we can select dataset like Breast Cancer,IRIS & Wine dataset and can 
also select classifier like KNN, Random Forest & SVM.
We can also update K(parameters) as per our convenience.
Whenever we change any of the above features accuracy will be updated accordingly.


## File details
- Dockerfile 
- requirement.txt
- Test.py


### Author Details:-
-  Kavita Joshi 
-  M20AIE244
-  joshi.14@iitj.ac.in 
  


