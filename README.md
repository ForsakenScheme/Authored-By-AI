# ABAI : Authored By AI

## Authored by AI : The authorship attribution tool

**Authorship attribution** is the task of identifying the author of a given text.

*Authored by AI* is a tool capable to **detect** whether a particular **text** has been **written by** a **human** author **or** whether it has been **generated by** a generative **AI** model (ChatGPT for instance). The tool itself is powered by a machine learning model, that was **trained on a labelled dataset of texts** written by human authors or generated by generative AI. Multiple supervised machine learning algorithms have been tested and evaluated during the development of *ABAI*'s model. 

## How to use it ?

Currently there is **two ways** to use the tool. 

    The **former** is **web-based** and can be started by using **python** in a terminal or **docker**(commands and GUI). 
    The **latter** is based on a local **PyQt5 GUI** which can **only be launched through python** in a terminal because of **GUI dependencies not being properly handled** in a docker environment. 

In the web-based version you only have access to pre-trained models that you can load. It is not possible to train, validate or test new models. 
Therefore you also cannot change any settings or modify the database. 

In the local PyQt5 GUI you will have access to the database, you will be able to delete, add and update texts present in the databases.
It's possible to train, validate and test multiple models you can choose from. You are also allowed to change feature extraction settings from the configuration window. Each step (training, validation, testing) can be performed individually. The training will always be based on the settings from the configuration window (also changeable in the config.ini file located at: code\backend\config\config.ini)
The menus aren't flawless but they do the job for now. 

## Docker (Web-based only)

### Docker commands only
#### Web-based
1. First **run** the command ```docker-compose -f docker/docker-compose.yml build``` from the 2324_INFOB318_ABAI2-[version number] current working directory.
2. Then **run** the command ```docker-compose -f docker/docker-compose.yml up abai-website``` still from the 2324_INFOB318_ABAI2 current working directory.
3. Open your favourite browser and **go to** : http://172.28.112.1:8000/ (accessible from any device in the local network) or http://localhost:8000/ (accessible from **local host** device **only**).
4. Copy and paste the text to be analysed in the input box on the web interface (10000 characters maximum).
5. Hit the "detect origin" button.
6. Check the results human/ai. 
   
### Docker Desktop GUI
#### Web-based
1. First **run** the command ```docker-compose -f docker/docker-compose.yml build``` from the 2324_INFOB318_ABAI2-[version number] current working directory.
2. Then on the Docker GUI chose the abai-web image and start the container with a name of your choice and specify the port 8000.
3. Open your favourite browser and **go to** : http://172.28.112.1:8000/ (accessible from any device in the local network) or http://localhost:8000/ (accessible from **local host** device **only**).
4. Copy and paste the text to be analysed in the input box on the web interface (10000 characters maximum).
5. Hit the "detect origin" button.
6. Check the results human/ai. 
   
### Standalone terminal (Python required)
#### Web-based
1. First **navigate** to the **2324_INFOB318_ABAI2-[version number] folder** and, inside the folder, **run** the command ```pip install -r requirements.txt```
2. Still inside the folder, **run** the command ```python -O .\code\django_abai\manage.py runserver 0.0.0.0:8000```
3. Open your favourite browser and **go to** : http://localhost:8000/. Optionnaly you can also access it from another device connected to the local network by using the ip address (run ipconfig in a terminal) of your host device.
4. Copy and paste the text to be analysed in the input box on the web interface (10000 characters maximum).
5. Hit the "detect origin" button.
6. Check the results human/ai. 
   
#### PyQt5 GUI
1. First **navigate** to the **2324_INFOB318_ABAI2-[version number] folder** and, inside the folder, **run** the command ```pip install -r requirements.txt```
2. Still inside the folder, **run** the command ```python -O .\code\backend\main.py```
3. A window will open, navigate to **Detect Origin**
4. A new window will open where you can select the model to be used and input the text to be analyzed.
5. Hit the "detect origin" button.
6. Check the results human/ai. 
   
### Disclaimer
The project is a WIP and started as a small University project. I'm working on it by myself during my freetime to gather experience. Some parts may not work as expected, please contact me and be patient. You can also contribute if you wish to. I'm open to every suggestion and help ! Don't hesitate to contact me for any reason. Thanks for your time reading this ! 

Remember **not to use ABAI as a primary decision-making tool**, but rather as a complementary method of determining the source of a writing. 
