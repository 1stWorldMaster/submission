Calls the tui for the application 

TUI 
 i) view previous data
 ii) run model
    video
        list of video using fzf allows to select
    live camera
    back
exit

view previous data 
 show them table which allow them to search too
 basic html page

main.py
    call the rapper function from the tui giving it options
    for the selected option run the model for inference

tui.py
    will define the rapper

model.py will give the rapper like 
    video_feed
    camera_feed
    
1. Build the model use transfer in process
	In this model will launch the other propcess for streaming the person and another process for processing of the person. 
	 	use cv2
	
2. Build store frame and run model
	will sotre the frame of the video and run the model on inference for every frame
		use cv2
	
3. write video and run
	use ffmpeg or cv to load the video of 10 secs and run the model inference on that model (checking frame or model is not updated upon certain time turn off the model)
		-use ffmpeg - video_ffmpeg
		-use cv2







proccall =  ffmpeg stream
while proc.proll is not None:
    if file_name is available
        file_name = (just increase the number)
            check normally for the video




download the raw package, in the same folder
downlaod using python package3.11
downlaod using apt

make the venv package
load the venv package

run the main.py


1. Check for raw package in the folder named package
2.  if not present downlaod else continue
3. Check if the venv is not present 
4. If not present load it then 
5. check for the packages if not presenet run for pip install -r requirenment.txt
6. Check if model is present or not
7. if model is present then load it and continue 
8. continue with the code by running python main.py




1. Check for downlaod package in the folder python-3.11.tar(maybe)
2. if not present downlaod it else continue
3. if not downloaded , download it using apt
4. now check for venv
5. if not present load the package and downlaod all the requirements
6. activate the venv
7. run python main.py