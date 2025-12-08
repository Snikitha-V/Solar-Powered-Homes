import requests
import cv2
import numpy as np

api_key = 'AIzaSyBXV54hr_1nkQqHPIBSQcO3e1mMvKvs1xI'
url = 'https://maps.googleapis.com/maps/api/staticmap'
params = {
    'center': '28.7041,77.1025',
    'zoom': 20,
    'size': '640x640',
    'maptype': 'satellite',
    'key': api_key
}

print('Testing Google Maps API...')
try:
    response = requests.get(url, params=params, timeout=30)
    print(f'Status: {response.status_code}')
    print(f'Content length: {len(response.content)}')
    print(f'Content type: {response.headers.get("content-type", "unknown")}')
    
    if response.status_code != 200:
        print(f'Error: {response.text[:500]}')
    else:
        print('✓ API call successful')
        
        # Try to decode image
        img_array = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if image is not None:
            print(f'✓ Image decoded successfully: {image.shape}')
        else:
            print('✗ Failed to decode image')
            
except Exception as e:
    print(f'Exception: {e}')
