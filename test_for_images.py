import requests
import time

# API endpoint
url = "https://db.ygoprodeck.com/api/v7/cardinfo.php"

# Request parameters
params = {
    "type": "Normal Monster"
}

# Number of images to download
num_images = 3

try:
    # Send GET request to the API
    response = requests.get(url, params=params)
    data = response.json()

    # Check if the request was successful
    if response.status_code == 200:
        # Retrieve the list of cards
        cards = data["data"]

        # Iterate over the cards and download images
        for i in range(num_images):
            card = cards[i]
            image_url = card["card_images"][0]["image_url_cropped"]
            image_name = f"image_{i}.jpg"

            # Download the image
            response = requests.get(image_url)

            # Check if the image download was successful
            if response.status_code == 200:
                # Save the image to disk
                with open(image_name, "wb") as file:
                    file.write(response.content)
                print(f"Image {i + 1} downloaded and saved.")

                # Wait for 2 seconds before downloading the next image
                time.sleep(2)

            else:
                print(f"Failed to download image {i + 1} with status code:", response.status_code)

    else:
        print("Request failed with status code:", response.status_code)

except requests.exceptions.RequestException as e:
    print("An error occurred:", e)
