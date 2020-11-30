
import pandas as pd
import numpy as np
import os
import sys
import xml.sax
import xml.parsers.expat
import time
import pickle
import os
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import OrderedDict
import re
# Data Loader
titles = []
labels = []
text=[]

class ParseHandler(xml.sax.handler.ContentHandler):
    def __init__(self):
        self.text=""
    def startElement(self, name, attrs):
        self.CurrentData = name
        if(self.CurrentData=="article"):
            labels.append(attrs['hyperpartisan'])


parser = xml.sax.make_parser()
parser.setFeature(xml.sax.handler.feature_namespaces, 0)
Handler = ParseHandler()
parser.setContentHandler(Handler)
parser.parse("gd.xml")


class ParseHandler1(xml.sax.handler.ContentHandler):
    def __init__(self):
        self.text=""
        self.r1=0
    def startElement(self, name, attrs):
        self.CurrentData = name
        if 'title' in attrs.getNames():
            titles.append(attrs.getValue('title'))
    def endElement(self, name):
        if(name=="article"):
            self.text=clean_text(self.text)
            self.text=re.sub(r'([^\s\w]|_)+', '', self.text)
            self.text=self.text.strip()
            text.append(self.text)
            self.text=""
    def characters(self, data):
        self.text += data

def clean_text(text1):
    text1 = text1.replace(".", ". ")
    text1 = text1.replace(" _", " ")
    text1 = text1.replace("  ", " ")
    text1 = text1.replace(". . . .", "...")
    text1 = text1.replace(". . .", "...")
    return text1

parser = xml.sax.make_parser()
parser.setFeature(xml.sax.handler.feature_namespaces, 0)
Handler = ParseHandler1()
parser.setContentHandler(Handler)
parser.parse("bypublisher.xml")


df_text = pd.DataFrame(text)
df_label = pd.DataFrame(labels)
df_title = pd.DataFrame(titles)
# print(df1)
result = pd.concat([df_text, df_title, df_label], axis=1, sort=False)
result.columns = ['texts','title','labels']
result.to_csv('publisher.csv')
