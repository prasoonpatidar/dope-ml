#!/usr/bin/python

cat_2014 = './annotations/instances_val2014.json'
cat_2017 = './annotations/instances_val2017.json'

import sys, getopt
import json

def main(argv):
    json_file = None
    try:
        opts, args = getopt.getopt(argv,"hy:")
    except getopt.GetoptError:
        print('coco_categories.py -y <year>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-y':
            if(arg == '2014'):
                json_file = cat_2014
            else:
                json_file = cat_2017
    if json_file is not None:
        with open(json_file,'r') as COCO:
            js = json.loads(COCO.read())
            return js['categories']

if __name__ == "__main__":
    categories = main(['-y','2017'])
    list_cats = [categories[kr]['name'] for kr in range(len(categories))]

    pass