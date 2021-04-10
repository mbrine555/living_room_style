import os
import time
import requests
from bs4 import BeautifulSoup

def get_page(url):
	try:
		print(f'Requesting {url}')
		response = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:78.0) Gecko/20100101 Firefox/78.0',
                                                          'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                                                          'Accept-Language': 'en-US,en;q=0.5',
                                                          'Accept-Encoding': 'gzip, deflate, br',
                                                          'TE': 'Trailers',
                                                          'Host': 'www.houzz.com',
                                                          'Upgrade-Insecure-Requests': '1',
                                                          'Cookie': 'v=1596493013_1a903bfa-909f-49cf-ab6c-42e83d59f2f5_b839bc669c17467060f0b3870b63f76c; vct=en-US-vxnVjChfyBvVjChfSBzVjChf; _csrf=9CQ3HxIBfgPzf7-wMpM70jSP; jdv=; documentWidth=1536; G_ENABLED_IDPS=google',
                                                          'DNT': '1',
                                                          'Connection': 'keep-alive'})
		response.raise_for_status()
	except requests.exceptions.HTTPError as err:
		print(err)
	
	page_content = BeautifulSoup(response.content, "html.parser")
	return page_content
	

def make_new_page_urls(url):
	page_content = get_page(url)
	total_results = page_content.find('span', {'class':'hz-top-pagination__text'}).find_all('b')[2].text
	total_results = int(total_results.replace(',', ''))
	
	new_urls = set([f'{url}?fi={x}' for x in range(18,min(total_results,1000),18)])
	return new_urls
	

def download_images(url, label):
	page_content = get_page(url)
	divs = page_content.find_all('div', {'class': 'hz-space-card hz-space-card-unify hz-track-me'})
	images = [div.find('img').get('src') for div in divs]
	image_urls = [image for image in images if 'https' in image]
	
	print(f'Downloading {len(image_urls)} images for {label}')
	os.makedirs(os.path.join('training', label), exist_ok=True)
	
	for i, url in enumerate(image_urls):
		pic_name = url.split('/')[-1]
		with open(f'training/{label}/{pic_name}', 'wb') as handle:
			response = requests.get(url, stream=True)
		
			if not response.ok:
				print(response)
		
			for block in response.iter_content(1024):
				if not block:
					break
			
				handle.write(block)


def download_style(base_url, label):
	new_urls = make_new_page_urls(base_url)
	new_urls.add(base_url)
	
	for url in new_urls:
		download_images(url, label)
		time.sleep(18)
		

if __name__ == '__main__':
				 
	base_urls = {
        'industrial': 'https://www.houzz.com/photos/industrial-living-room-ideas-phbr1-bp~t_718~s_2113',
		'victorian': 'https://www.houzz.com/photos/victorian-living-room-ideas-phbr1-bp~t_718~s_22849',
		'shabby-chic': 'https://www.houzz.com/photos/shabby-chic-style-living-room-ideas-phbr1-bp~t_718~s_22847',
		'country': 'https://www.houzz.com/photos/farmhouse-living-room-ideas-phbr1-bp~t_718~s_2114',
		'contemporary': 'https://www.houzz.com/photos/contemporary-living-room-ideas-phbr1-bp~t_718~s_2103',
		'traditional': 'https://www.houzz.com/photos/traditional-living-room-ideas-phbr1-bp~t_718~s_2107',
		'modern': 'https://www.houzz.com/photos/modern-living-room-ideas-phbr1-bp~t_718~s_2105',
		'mid-century': 'https://www.houzz.com/photos/midcentury-modern-living-room-ideas-phbr1-bp~t_718~s_2115',
		'mediterranean': 'https://www.houzz.com/photos/mediterranean-living-room-ideas-phbr1-bp~t_718~s_2109',
		'scandanavian': 'https://www.houzz.com/photos/scandinavian-living-room-ideas-phbr1-bp~t_718~s_22848',
		'asian': 'https://www.houzz.com/photos/asian-living-room-ideas-phbr1-bp~t_718~s_2102',
		'southwestern': 'https://www.houzz.com/photos/southwestern-living-room-ideas-phbr1-bp~t_718~s_14159'
	}
	
	
	for label, base_url in base_urls.items():
		download_style(base_url, label)
		