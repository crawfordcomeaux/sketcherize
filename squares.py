import sys
import numpy
from PIL import Image, ImageOps, ImageDraw
from scipy.ndimage import morphology, label
from operator import itemgetter, attrgetter
# from rtree import index
# import cv2 as cv

def boxes(orig):
    # SciPy
    img = ImageOps.grayscale(orig)
    # OpenCV
    # img = cv.cvtColor(orig,cv.COLOR_RGB2GRAY)

    im = numpy.array(img)

    # Inner morphological gradient.
    #
    # The SciPy way
    im = morphology.grey_dilation(im, (3, 3)) - im
    # The OpenCV way
    # im2 = cv.dilate(im, None) - im

    # Binarize.
    mean, std = im.mean(), im.std()
    t = mean + std
    im[im < t] = 0
    im[im >= t] = 1

    # Connected components.
    lbl, numcc = label(im)
    # Size threshold.
    min_size = 200 # pixels
    box = []
    for i in range(1, numcc + 1):
        py, px = numpy.nonzero(lbl == i)
        if len(py) < min_size:
            im[lbl == i] = 0
            continue

        xmin, xmax, ymin, ymax = px.min(), px.max(), py.min(), py.max()
        box.append(Rectangle(xmin, ymin, xmax, ymax))
        # Four corners and centroid.
        # box.append({'points': [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)], 'centroid': (numpy.mean(px), numpy.mean(py)), 'width':xmax - xmin, 'height':ymax - ymin})
        # box.append({'p1': (xmin, ymin), 'p2': (xmax, ymax), 'centroid': (numpy.mean(px), numpy.mean(py)), 'width':xmax - xmin, 'height':ymax - ymin})
    '''
    children_processed = []
    nested = []
    for b in box:
        print "----"
        if b in children_processed:
            print "Time to skip"
            skip = True
            continue
        else:
            skip = False
            print "Don't skip"
        if skip:
            print "Failed to skip"
        children = [x for x in box if (b['p1'] != x['p1'] and (b['width'] * b['height']) > (x['width'] * x['height']) and b['p2'] != x['p2'] and b['p1'][0] < x['p2'][0] and b['p2'][0] > x['p1'][0] and b['p1'][1] < x['p2'][1] and b['p2'][1] > x['p1'][1])]

        children_processed += children
        if children:
            b.update({'children':children})
        nested.append(b)
    '''
    return im.astype(numpy.uint8) * 255, box

class Rectangle(object):
    def __init__(self, x1, y1, x2, y2):
        self.p1 = (x1, y1)
        self.p2 = (x2,y2)
        self.children = []

    def __str__(self):
        return "Rectangle defined by %s, %s, %i children" % (self.p1, self.p2, len(self.children))
    def is_child_of(self, other):
        return (self is not other and
                self not in other.children and
            ((self.p2[0] - self.p1[0]) * (self.p2[1] - self.p1[1])) < ((other.p2[0] - other.p1[0]) * (other.p2[1] - other.p1[1])) and
            self.p1[0] > other.p1[0] and
            self.p2[0] < other.p2[0] and
            self.p1[1] > other.p1[1] and
            self.p2[1] < other.p2[1])

    def add_child(self, other):
        self.children.append(other)

    def check_relation_and_connect(self, other):
        if self.is_child_of(other):
            other.add_child(self)
        elif other.is_child_of(self):
            self.add_child(other)

def insertRect(newRect, rectList):
    hasBeenInserted = False
    for r in rectList:
        if newRect.is_child_of(r):
            if not r.children:
                hasBeenInserted = True
                r.add_child(newRect)
                return hasBeenInserted
            else:
                hasBeenInserted = insertRect(newRect, r.children)
    if not hasBeenInserted:
        rectList.append(newRect)
        hasBeenInserted = True
    return hasBeenInserted

def buildTreeFromList(rectList):
    tree = []
    for r in rectList:
        insertRect(r,tree)
    return tree

def buildFragmentFromTree(rectTree, depth=0):
    fragment = ''
    indent = '\t' * depth
    for r in rectTree:
        fragment += '\n' + indent + '<div data-p1="' + str(r.p1) + '" data-p2="' + str(r.p2) + '">'
        #fragment += '\n' + indent + '<div>'
        if r.children:
            fragment += buildFragmentFromTree(r.children, depth+1) + '\n'
        fragment += '</div>'
    return fragment

orig = Image.open("1.small.jpg")
im, box = boxes(orig)

tree = []
for b in tree:
  print b
buildTreeFromList(box)
tree = buildTreeFromList(box)
len(tree)
print buildFragmentFromTree(tree)
print box[0]
len(box)
for b in box:
  print b not in box[0].children

print box[0]
f = itemgetter(0)
test = sorted(box, key=lambda x: (x.p1[1],x.p1[0]))
print test
for t in tree[0].children:
  print t
a= buildFragmentFromTree(tree)
b = buildFragmentFromTree(box)
c =buildFragmentFromTree(test)
a == b
b == c
a == c
print len(a)
print b
print c

# Boxes found.
Image.fromarray(im).save("1.small.boxes.jpg")

# Draw perfect rectangles and the component centroid.
img = Image.fromarray(im)
visual = img.convert('RGB')
draw = ImageDraw.Draw(visual)

#def gen_rtree(boxes):
#    for i, box in enumerate(boxes):
#        yield (i, (box[0], box[1], box[2], box[3]), box)
#r = index.Index(gen_rtree(box))



#for b, centroid in box:
#    draw.line(b + [b[0]], fill='yellow')
#    cx, cy = centroid
#    draw.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), fill='red')

#visual.save(sys.argv[3])

