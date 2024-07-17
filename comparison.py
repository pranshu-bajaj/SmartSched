import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

yolov_modelv3_weights = "./yolo-coco/yolov3.weights"
yolov_modelv3_configuration = "./yolo-coco/yolov3.cfg"
yolov_model = cv2.dnn.readNet(yolov_modelv3_weights, yolov_modelv3_configuration)

coco_n = "./yolo-coco/coco.names"
classes = []
with open(coco_n,'r') as f:
  classes = f.read().splitlines()



def count(index):
    img = cv2.imread(f"./Dataset-1/images/{index}.jpg")
    height , width, _ = img.shape
    plt.imshow(img)

    # converting image to binary large object
    blob = cv2.dnn.blobFromImage(img,1/255,(320,320),(0,0,0),swapRB=True,crop=False)

    i= blob[0].reshape(320,320,3)
    plt.imshow(i)

    yolov_model.setInput(blob)

    output_layers_name = yolov_model.getUnconnectedOutLayersNames()
    # forward pass
    layer_output = yolov_model.forward(output_layers_name)

    boxes = []
    confidences = []
    class_ids = []

    for output in layer_output:
      for detection in output:
        score = detection[5:]
        class_id = np.argmax(score)
        confidence = score[class_id]
        if confidence > 0.7:
          center_x = int(detection[0]*width)
          center_y = int(detection[1]*height)
          w = int(detection[2]*width)
          h = int(detection[3]*height)

          x = int(center_x-w/2)
          y = int(center_y-h/2)

          boxes.append([x,y,w,h])
          confidences.append((float(confidence)))
          class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes ,confidences , 0.5 , 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0,255,size=(len(boxes),3))

    cnt_1 = 0
    cnt_2 = 0
    if len(indexes) > 0:
      for i in indexes.flatten():
        x,y,w,h = boxes[i]
        label = str(classes[class_ids[i]])
        confi = str(round(confidences[i],3))
        color = colors[i]
        if(label == "bicycle" or label == "car" or label == "motorbike" or label == "bus" or label == "truck"):
          if(x<width/2):
            cnt_1 = cnt_1+1
          else:
            cnt_2 = cnt_2+1
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        cv2.putText(img,label + " " + confi,(x,y+20),font,2,(1,1,1),2)

    # can uncomment to see the image
    # cv2.imshow("image",img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    l = [cnt_1,cnt_2]
    return l



def max_comb(vehicle_count,wait):
    max_score=-1e9
    mx_lanes=[]
    check=[1,3,4]
    for i in range(0,8,2):
        for j in check:
            score=vehicle_count[i]*wait[i]+vehicle_count[(i+j)%8]*wait[(i+j)%8]
            if score>max_score:
                max_score=score
                mx_lanes=[i,(i+j)%8]
    return mx_lanes



def update(vehicle_count,wait,cur_1,cur_2):
    even = [1,3,4]
    odd = [-1,5]
    mx = -1e9
    new_1,new_2 = cur_1,cur_2

    parity_1 = even if (cur_1&1)==0 else odd
    for i in parity_1:
        a = (cur_1 + i)%8
        if a==cur_2:
            continue
        score = vehicle_count[a] * wait[a] + vehicle_count[cur_1] * wait[cur_1]
        if score > mx:
            mx = score
            new_1 = cur_1
            new_2 = a

    if cur_2!=-1:
      parity_2 = even if (cur_2&1)==0 else odd
      for i in parity_2:
          a = (cur_2 + i)%8
          if a==cur_1:
              continue
          score = vehicle_count[a] * wait[a] + vehicle_count[cur_2] * wait[cur_2]
          if score > mx:
              mx = score
              new_1 = a
              new_2 = cur_2
    return new_1,new_2


def algo_1(vehicle_count,wait,cnt,cur_1,cur_2):
    t=0
    while cnt > 0:
        # print(cur_1,cur_2)
        c = min(vehicle_count[cur_1],vehicle_count[cur_2])
        if c==0:
           if(vehicle_count[cur_1]==vehicle_count[cur_2]):
               cur_1,cur_2 = max_comb(vehicle_count,wait)
               continue
           else:
               c = max(vehicle_count[cur_1],vehicle_count[cur_2])
               cnt -= c
               vehicle_count[cur_1] = 0
               vehicle_count[cur_2] = 0
        else:
            vehicle_count[cur_1] -= c
            vehicle_count[cur_2] -= c
            cnt = cnt - 2*c
        t +=c

        wait[cur_1] -= 1
        wait[cur_2] -= 1
        # updating current lanes
        cur_1,cur_2 = update(vehicle_count,wait,cur_1,cur_2)
        # print("time:",t)
        # print("vehicles remaining:",cnt)
    return t



def algo_2(vehicle_count,wait,cnt,cur_1,cur_2):
    t=0
    while cnt > 0:
        # print(cur_1,cur_2)
        c = min(vehicle_count[cur_1],vehicle_count[cur_2])
        if c==0:
           if(vehicle_count[cur_1]==vehicle_count[cur_2]):
               cur_1,cur_2 = max_comb(vehicle_count,wait)
               continue
           else:
               c = max(vehicle_count[cur_1],vehicle_count[cur_2])
               cnt -= c
               vehicle_count[cur_1] = 0
               vehicle_count[cur_2] = 0
        else:
            vehicle_count[cur_1] -= c
            vehicle_count[cur_2] -= c
            cnt = cnt - 2*c
        t +=c

        wait[cur_1] -= 1
        wait[cur_2] -= 1
        # updating current lanes
        cur_1,cur_2 = max_comb(vehicle_count,wait)
        # print("time:",t)
        # print("vehicles remaining:",cnt)
    return t



cycles = 1
n = 10
total_time_1 = 0
total_time_2 = 0
total_vehicles = 0
while cycles <= n:
    vehicle_count = []
    vehicle_count_2=[]
    for i in range(4):
        a = random.randint(0,1200)
        vehicle_count += count(a)
        vehicle_count_2 += count(a)

    # vehicle_count = [3,2,5,5,2,3,5,1]
    print("\nCycle",cycles,"\n")
    print("Count of vehicles in each lane:",vehicle_count)
    wait = [8,8,8,8,8,8,8,8]
    wait_2 = [8,8,8,8,8,8,8,8]
    cnt = sum(vehicle_count)
    cnt_2 = sum(vehicle_count_2)
    total_vehicles += cnt
    print("Total Vehicles:",cnt)

    cur_1,cur_2=max_comb(vehicle_count,wait)
    t1=algo_1(vehicle_count,wait,cnt,cur_1,cur_2)
    total_time_1 += t1
    cur_1,cur_2=max_comb(vehicle_count,wait)
    t2=algo_2(vehicle_count_2,wait_2,cnt_2,cur_1,cur_2)
    total_time_2 += t2
    print("Time taken by Paper Algorithm:",t1)
    print("Time taken by Updated Algorithm:",t2,"\n")
    cycles += 1

print("\nNo of vehicles:",total_vehicles)
print("Total Time taken by Paper Algorithm:",total_time_1)
print("Total Time taken by Updated Algorithm:",total_time_2,"\n")