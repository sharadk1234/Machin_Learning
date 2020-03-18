{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting scikit-image\n",
      "  Using cached scikit_image-0.16.2-cp37-cp37m-macosx_10_6_intel.whl (30.3 MB)\n",
      "Requirement already satisfied: matplotlib!=3.0.0,>=2.0.0 in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from scikit-image) (3.1.1)\n",
      "Collecting imageio>=2.3.0\n",
      "  Downloading imageio-2.8.0-py3-none-any.whl (3.3 MB)\n",
      "\u001b[K     |████████████████████████████████| 3.3 MB 2.5 MB/s eta 0:00:01\n",
      "\u001b[?25hCollecting PyWavelets>=0.4.0\n",
      "  Using cached PyWavelets-1.1.1-cp37-cp37m-macosx_10_9_x86_64.whl (4.3 MB)\n",
      "Requirement already satisfied: pillow>=4.3.0 in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from scikit-image) (7.0.0)\n",
      "Collecting networkx>=2.0\n",
      "  Using cached networkx-2.4-py3-none-any.whl (1.6 MB)\n",
      "Requirement already satisfied: scipy>=0.19.0 in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from scikit-image) (1.3.1)\n",
      "Requirement already satisfied: kiwisolver>=1.0.1 in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from matplotlib!=3.0.0,>=2.0.0->scikit-image) (1.1.0)\n",
      "Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from matplotlib!=3.0.0,>=2.0.0->scikit-image) (2.4.2)\n",
      "Requirement already satisfied: python-dateutil>=2.1 in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from matplotlib!=3.0.0,>=2.0.0->scikit-image) (2.8.0)\n",
      "Requirement already satisfied: numpy>=1.11 in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from matplotlib!=3.0.0,>=2.0.0->scikit-image) (1.17.2)\n",
      "Requirement already satisfied: cycler>=0.10 in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from matplotlib!=3.0.0,>=2.0.0->scikit-image) (0.10.0)\n",
      "Requirement already satisfied: decorator>=4.3.0 in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from networkx>=2.0->scikit-image) (4.4.0)\n",
      "Requirement already satisfied: setuptools in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from kiwisolver>=1.0.1->matplotlib!=3.0.0,>=2.0.0->scikit-image) (40.8.0)\n",
      "Requirement already satisfied: six>=1.5 in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from python-dateutil>=2.1->matplotlib!=3.0.0,>=2.0.0->scikit-image) (1.12.0)\n",
      "Installing collected packages: imageio, PyWavelets, networkx, scikit-image\n",
      "Successfully installed PyWavelets-1.1.1 imageio-2.8.0 networkx-2.4 scikit-image-0.16.2\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install scikit-image"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#!/usr/bin/env python\n",
    "# coding: utf-8\n",
    "\n",
    "# In[8]:\n",
    "\n",
    "\n",
    "import numpy\n",
    "import json\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import pandas as pd\n",
    "from collections import Counter\n",
    "\n",
    "def filtered_var(original,to_be_filter):\n",
    "    for each in to_be_filter:\n",
    "        original.remove(each)\n",
    "        \n",
    "    return original\n",
    "    \n",
    "def draw_graph(values):\n",
    "    plt.style.use('ggplot')\n",
    "    plt.hist(values, bins=100)\n",
    "    plt.show()\n",
    "    \n",
    "    return\n",
    "\n",
    "def get_bounds(variable):\n",
    "\tvalues = variable\n",
    "\tvalues.sort()\n",
    "\tlength = len(values)\n",
    "\tinedx= int((length - 1)/4)\n",
    "\tq1 = (values[inedx] + values[inedx + 1])/2\n",
    "\tinedx = int((length - 1)*3/4)\n",
    "\tq3 = (values[inedx] + values[inedx + 1])/2\n",
    "\tiqr = q3 - q1\n",
    "\tlower_bound = q1 - 1.5*iqr            #100\n",
    "\tupper_bound = q3 + 1.5*iqr\n",
    "\treturn lower_bound, upper_bound\t\n",
    "\n",
    "def get_outlier(variable):\n",
    "    values = list(variable.values())\n",
    "    lower_bound, upper_bound = get_bounds(values)\n",
    "    outlier = []\n",
    "    for i in values:\n",
    "        if (i > upper_bound or i <lower_bound):\n",
    "            outlier.append(i)\n",
    "    return outlier\n",
    "\n",
    "with open('train.json') as f:\n",
    "    data = json.load(f)\n",
    "    \n",
    "    price = data['price']\n",
    "latitude = data['latitude']\n",
    "longitude = data['longitude']\n",
    "time = data['created']\n",
    "\n",
    "new_time = time\n",
    "for keys in time:\n",
    "    new_time[keys] = time[keys][11] + time[keys][12]\n",
    "\n",
    "for keys in new_time:\n",
    "    tmp = int(new_time[keys])\n",
    "    new_time[keys] = tmp\n",
    "    \n",
    "draw_graph(list(price.values()))\n",
    "draw_graph(list(latitude.values()))\n",
    "draw_graph(list(longitude.values()))\n",
    "\n",
    "x= list(time.values())\n",
    "x.sort()\n",
    "plt.xlim(1,24)\n",
    "plt.style.use('ggplot')\n",
    "plt.hist(x, bins=24)\n",
    "plt.xticks(range(1,24))\n",
    "plt.show()\n",
    "\n",
    "top_5_hours = dict(Counter(new_time.values()).most_common(5))\n",
    "other_hours = len(new_time) - sum(top_5_hours.values())\n",
    "proportion={'other hours':other_hours}\n",
    "for each in top_5_hours:\n",
    "    proportion.update({each:top_5_hours[each]})\n",
    "\n",
    "plt.pie(proportion.values(), labels=proportion.keys(),autopct='%1.1f%%', shadow=True, startangle=140)\n",
    "plt.axis('equal')\n",
    "plt.show()\n",
    "\n",
    "df = pd.DataFrame(data)\n",
    "outlier_price= get_outlier(price)\n",
    "outlier_latitude = list(df[(df['latitude'].map(lambda d: d < 40))]['latitude'])\n",
    "outlier_longitude = list(df[(df['longitude'].map(lambda d: d < -79.8 or d > -73.3))]['longitude'])\n",
    "\n",
    "missing_value = df[(df['features'].map(lambda d: len(d)) == 0) | (df['display_address'].map(lambda d: d) == '') |\n",
    "                    (df['description'].map(lambda d: d) == '') | (df['building_id'].map(lambda d: d) == '0') | (df['latitude'].map(lambda d: d) == 0) | (df['longitude'].map(lambda d: d) == 0)]\n",
    "missing_feature_number = df[(df['features'].map(lambda d: len(d)) == 0)]['features'].count()\n",
    "missing_display_address_number = df[(df['display_address'].map(lambda d: d) == '')]['display_address'].count()\n",
    "missing_description_number = df[df['description'].map(lambda d: d) == '']['description'].count()\n",
    "missing_building_id_number = df[df['building_id'].map(lambda d: d) == '0']['building_id'].count()\n",
    "missing_latitude_number = df[(df['latitude'].map(lambda d: d) == 0)]['latitude'].count()\n",
    "missing_longitude_number = df[(df['longitude'].map(lambda d: d) == 0)]['longitude'].count()\n",
    "missing_street_address_number = df[(df['street_address'].map(lambda d: d) == '')]['street_address'].count()\n",
    "\n",
    "filter_price = filtered_var(list(price.values()),outlier_price)\n",
    "filter_latitude = filtered_var(list(latitude.values()),outlier_latitude)\n",
    "filter_longitude = filtered_var(list(longitude.values()),outlier_longitude)\n",
    "\n",
    "draw_graph(filter_price)\n",
    "draw_graph(filter_latitude)\n",
    "draw_graph(filter_longitude)\n",
    "\n",
    "plt.boxplot(list(price.values()))\n",
    "plt.show()\n",
    "plt.boxplot(filter_price)\n",
    "plt.show()\n",
    "plt.boxplot(list(latitude.values()))\n",
    "plt.show()\n",
    "plt.boxplot(filter_latitude)\n",
    "plt.show()\n",
    "plt.boxplot(list(longitude.values()))\n",
    "plt.show()\n",
    "plt.boxplot(filter_longitude)\n",
    "plt.show()\n",
    "plt.scatter(range(1,len(price)+1),list(price.values()),alpha=0.5)\n",
    "plt.show()\n",
    "\n",
    "from sklearn.feature_extraction.text import CountVectorizer\n",
    "from sklearn.feature_extraction.text import TfidfTransformer\n",
    "\n",
    "feature_content = []\n",
    "for i in data['features'].values():\n",
    "    feature_content += i\n",
    "    \n",
    "count_vect = CountVectorizer(ngram_range=(1,2), analyzer='word')\n",
    "x_train_counts = count_vect.fit_transform(feature_content)\n",
    "\n",
    "tfidf_transformer = TfidfTransformer()\n",
    "x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)\n",
    "total_count = 0\n",
    "for each in count_vect.vocabulary_.values():\n",
    "    total_count = total_count + each\n",
    "\n",
    "feature_frequency_df = pd.DataFrame(x_train_tfidf.todense(), columns=count_vect.get_feature_names())\n",
    "\n",
    "frequency_feature = {}\n",
    "for keys in count_vect.vocabulary_.keys():\n",
    "    frequency_feature.update({keys:float((count_vect.vocabulary_.get(keys)/total_count))})\n",
    "\n",
    "\n",
    "# In[9]:\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "# In[ ]:\n",
    "\n",
    "\n",
    "from PIL import Image\n",
    "import requests\n",
    "from io import BytesIO\n",
    "from skimage import exposure\n",
    "from skimage import feature\n",
    "\n",
    "Hog_descriptor = data['photos']\n",
    "\n",
    "for each1 in Hog_descriptor:\n",
    "    descriptor_list = []\n",
    "    for each2 in Hog_descriptor.get(each1):\n",
    "        try:\n",
    "            response = requests.get(each2)\n",
    "            img = Image.open(BytesIO(response.content))\n",
    "            width, height = img.size  \n",
    "            left = 4\n",
    "            top = height / 5\n",
    "            right = 154\n",
    "            bottom = 3 * height / 5\n",
    "            newsize = (round(width/2), round(height/2))\n",
    "            im1 = img.crop((left, top, right, bottom))\n",
    "            im1 = im1.resize(newsize)\n",
    "            H = feature.hog(im1, orientations=9, pixels_per_cell=(1, 1),\n",
    "                cells_per_block=(2, 2), transform_sqrt=True, block_norm=\"L1\")\n",
    "            descriptor_list.append(H)\n",
    "        except (OSError, NameError):\n",
    "            print(each2)\n",
    "    Hog_descriptor.update({each1:descriptor_list})\n",
    "\n",
    "\n",
    "# In[25]:\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "# In[24]:\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "# In[6]:\n",
    "\n",
    "\n",
    "outlier_price= get_outlier(price)\n",
    "outlier_latitude = list(df[(df['latitude'].map(lambda d: d < 40))]['latitude'])\n",
    "outlier_longitude = list(df[(df['longitude'].map(lambda d: d < -79.8 or d > -73.3))]['longitude'])\n",
    "\n",
    "\n",
    "# In[21]:\n",
    "\n",
    "\n",
    "outlier_price\n",
    "\n",
    "\n",
    "# In[20]:\n",
    "\n",
    "\n",
    "pd.DataFrame(data)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
