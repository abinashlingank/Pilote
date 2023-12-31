import math
import numpy as np


def GetDistance(box, frame_dimensions, gsd):

    middle_frame = (frame_dimensions[0]//2, frame_dimensions[1]//2)

    middle_box = ((box[0]+box[0]+box[3]+box[4])/4, (box[1]+box[1]+box[2]+box[2])/4 )

    distance = (((middle_box[0]-middle_frame[0])**2)+((middle_box[1]-middle_frame[1])**2))**(0.5)

    real_distance = (gsd * distance) / 100000

    ph_end_x , ph_end_y = frame_dimensions[0], frame_dimensions[1]
    quad_1_dis = (((middle_box[0]-ph_end_x)**2)+((middle_box[1])**2))**(0.5)
    quad_2_dis = (((middle_box[0])**2)+((middle_box[1])**2))**(0.5)
    quad_3_dis = (((middle_box[0])**2)+((middle_box[1]-ph_end_y)**2))**(0.5)
    quad_4_dis = (((middle_box[0]-ph_end_x)**2)+((middle_box[1]-ph_end_y)**2))**(0.5)

    #Finding Angle
    y_distance = (((middle_box[0]-middle_frame[0])**2))**(0.5)
    angle = np.arcsin(y_distance/distance)        

    #Finding Quadrant
    dis_lst=[]
    dis_lst.append(quad_1_dis)
    dis_lst.append(quad_2_dis)
    dis_lst.append(quad_3_dis)
    dis_lst.append(quad_4_dis)

    if min(dis_lst) == quad_1_dis:
        # up_angle = angle
        heading_angle = (90 - angle) 
        quad = 'Quadrant 1'   
    elif min(dis_lst) == quad_2_dis:      
        # up_angle = -angle
        heading_angle =  -(90 - angle)
        quad = 'Quadrant 2'
    elif min(dis_lst) == quad_3_dis:   
        # up_angle = -(90+angle)
        heading_angle = (180 - angle)
        quad = 'Quadrant 3'
    elif min(dis_lst) == quad_4_dis:
        # up_angle = 90+angle
        heading_angle = -(180 - angle)
        quad = 'Quadrant 4'

    return real_distance, heading_angle

    


