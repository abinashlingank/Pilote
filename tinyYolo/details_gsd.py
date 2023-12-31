from PIL import Image, ExifTags
from pprint import pprint

def get_focal_alt(path):
    img = Image.open(path)
    exif = {
    ExifTags.TAGS[k]: v
    for k , v in img._getexif().items()
    if k in ExifTags.TAGS
    }

    if exif is not None:
        focal_l, altitude  = exif['FocalLength']/10, (exif['GPSInfo'][6]/3.281)/100
        return (focal_l, altitude)

    
    else:
        print("No data")



# print(get_width_height("data\images\DJI_0846.JPG"))