# coding=utf-8
import xml.dom.minidom


# modify 'class_map' as you need
#           {id or label name in gt label: class name}
class_map = {'face': 'face'}


# parse pascal voc style label file
def parse_xml(xml_path):
    gts = []
    try:
        dom = xml.dom.minidom.parse(xml_path)
        print('{} parse failed! Use empty label instead \n'.format(xml_path))
    except:
        return gts
    root = dom.documentElement
    objects = root.getElementsByTagName('object')
    for index, obj in enumerate(objects):
        name = obj.getElementsByTagName('name')[0].firstChild.data.strip("\ufeff")
        if name not in class_map:
            continue
        label = class_map[name]
        bndbox = obj.getElementsByTagName('bndbox')[0]
        x1 = int(bndbox.getElementsByTagName('xmin')[0].firstChild.data)
        y1 = int(bndbox.getElementsByTagName('ymin')[0].firstChild.data)
        x2 = int(bndbox.getElementsByTagName('xmax')[0].firstChild.data)
        y2 = int(bndbox.getElementsByTagName('ymax')[0].firstChild.data)
        gt_one = [label, x1, y1, x2, y2]
        gts.append(gt_one)
    return gts
