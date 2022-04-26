**<h1>Driver drowsiness detection</h1>**
Detecting and monitoring the driver's eyes (open/closed), mouth (yawn), and upper body (moving) trough camera. The program gives an alert sound, when the driver's eyes are closed for a given time (approximately 3-5 seconds). The whole detection, and monitoring is logged and the logs are saved into MySQL Database. When there is no alert happening, the saved log content from the database is deleted every 30 seconds, and only saved permanently if an alert happened.
_____________________________________________________
**<h2>Requirements:</h2>**
- Python                  >= 3.9     <br />
- Conda                   >= 4.10.3  <br />
- cmake                   >= 3.22.0  <br />
- dlib                    >= 19.22.0 <br />
- keras                   >= 2.7.0   <br />
- matplotlib              >= 3.5.0   <br />
- mysql-connector-python  >= 8.0.27  <br />
- numpy                   >= 1.21.4  <br />
- opencv-python           >= 4.5.4.60<br />
- Pillow                  >= 8.4.0   <br />
- pygame                  >= 2.1.0   <br />
- pyparsing               >= 3.6.0   <br />
- pyYaml                  >= 6.0     <br />
- scipy                   >= 1.7.2   <br />
- tensorboard             >= 2.7.0   <br />
- tensorflow              >= 2.7.0   <br />
- MySQL Community Edition >= 8.0.27  <br />
_____________________________________________________
**<h2>How to run my assignment:</h2>**
*/For IDE I used Pycharm/* <br />
1.  Install Anaconda for Windows / GithubDesktop / MySQL Community Edition / Pycharm
2.  Open Anaconda Prompt 3
3.  Run:<br /> `conda create -n myenv python=3.9`
4.  Activate environment with running this code:<br />`conda activate myenv`
5.  Run:<br />`cd myenv`
6.  Run:<br />`conda install -c conda-forge dlib`
7.  Create a requirements.txt file inside the virtual environment folder and copy paste the content from the `requirements.txt` you can find **inside this repository**     to the txt file you just created
8.  Run in the Anaconda Prompt 3:<br />`pip install -r requirements.txt` 
9.  Open GithubDesktop and click on Current Repository -> Clone Repository -> URL -> https://www.github.com/LeviG9901/DrowsinessDetection and then give a destination         where you want the repository to be cloned
