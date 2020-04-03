mkdir /home/fizzer/enph353_comp/src/competition_2019t2/src/vision/
mkdir /home/fizzer/enph353_comp/src/competition_2019t2/src/drive/
find /home/fizzer/enph353/vision/ -name \*.py -exec cp {} /home/fizzer/enph353_comp/src/competition_2019t2/src/vision/ \;
find /home/fizzer/enph353/drive -name \*.py -exec cp {} /home/fizzer/enph353_comp/src/competition_2019t2/src/drive/ \;
find /home/fizzer/enph353/ -name \*]main.py -exec cp {} /home/fizzer/enph353_comp/src/competition_2019t2/src/ \;
find /home/fizzer/enph353_comp/src/competition_2019t2/src/ -name \*.py -exec chmod +x {} \;
cd /home/fizzer/enph353_comp
catkin_make
