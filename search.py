import argparse
import requests
import os
import cv2
from requests import exceptions

# Argument parser to accept query and output directory
ap = argparse.ArgumentParser()
ap.add_argument("-q", "--query", required=True, help="search query to search Bing Image API for")
ap.add_argument("-o", "--output", required=True, help="path to output directory of images")
args = vars(ap.parse_args())

API_KEY = "d8982f9e69a4437fa6e10715d1ed691d"
MAX_RESULTS = 500
GROUP_SIZE = 50
URL = "https://api.cognitive.microsoft.com/bing/v7.0/images/search"
EXCEPTIONS = set([IOError, FileNotFoundError, exceptions.RequestException, 
                  exceptions.HTTPError, exceptions.ConnectionError, exceptions.Timeout])

# Create the output directory if it doesn't exist
os.makedirs(args["output"], exist_ok=True)

# Set the search term and headers for the Bing API request
term = args["query"]
headers = {"Ocp-Apim-Subscription-Key": API_KEY}
params = {"q": term, "offset": 0, "count": GROUP_SIZE}

# Search for images using Bing Image Search API
print("[INFO] Searching Bing API for '{}'...".format(term))
search = requests.get(URL, headers=headers, params=params)
search.raise_for_status()

# Get the total number of results and set the limit
results = search.json()
estNumResults = min(results["totalEstimatedMatches"], MAX_RESULTS)
print("[INFO] {} total results found for '{}'".format(estNumResults, term))

total = 0
for offset in range(0, estNumResults, GROUP_SIZE):
    print("[INFO] Requesting images for group {}-{} of {}...".format(offset, offset + GROUP_SIZE, estNumResults))
    
    # Update parameters for each group of results
    params["offset"] = offset
    search = requests.get(URL, headers=headers, params=params)
    search.raise_for_status()
    results = search.json()

    print("[INFO] Saving images for group {}-{}...".format(offset, offset + GROUP_SIZE))
    
    # Iterate over the results and download images
    for v in results["value"]:
        try:
            print("[INFO] Fetching image from: {}".format(v["contentUrl"]))
            r = requests.get(v["contentUrl"], timeout=30)
            r.raise_for_status()

            # Extract image extension and save path
            ext = v["contentUrl"][v["contentUrl"].rfind("."):]
            image_path = os.path.sep.join([args["output"], "{}{}".format(str(total).zfill(8), ext)])

            # Save image to file
            with open(image_path, "wb") as f:
                f.write(r.content)
            
            # Validate the image using OpenCV
            image = cv2.imread(image_path)
            if image is None:
                print("[INFO] Deleting invalid image: {}".format(image_path))
                os.remove(image_path)
                continue

            total += 1
        except Exception as e:
            if type(e) in EXCEPTIONS:
                print("[INFO] Skipping invalid image URL: {}".format(v["contentUrl"]))
                continue

print("[INFO] Downloaded {} images.".format(total))
