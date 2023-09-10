from bs4 import BeautifulSoup
import requests
import requests_cache
import tqdm

import re
import json

import time

requests_cache.install_cache('cache/',
                             backend='filesystem')

def get_soup(url):
    req = requests.get(url)
    return BeautifulSoup(req.text, "html.parser")

def get_smiles(cas):
    url = f'http://www.thegoodscentscompany.com/opl/{cas}.html'
    soup = get_soup(url)
    elem = soup.find(string=re.compile('SMILES :')).parent
    return elem.text.removeprefix('SMILES :')

def get_cas(soup):
    return soup.find('td',{'class':'radw11'}).text

def get_notes(soup):
    note_links = soup.find_all('a', {'href': re.compile(r'http://www\.thegoodscentscompany\.com/odor/')})
    return [r.text for r in note_links]

def get_blenders(soup):
    table_section = soup.find(string='Potential Blenders and core components ').parent.find_next('table')
    results = table_section.find_all(attrs={'class':'wrd80'})
    all_results = []
    for res in results:
        blender = res.find('a').text
        category = res.parent.find_previous(attrs={'class':'radw47'}).text
        all_results.append((blender,category))
    return all_results


def get_data(url):
    soup = get_soup(url)
    name, _, cas = soup.title.text.rpartition(', ')
    smiles = get_smiles(cas)
    notes = get_notes(soup)
    blenders = get_blenders(soup)
    return {'name':name,'cas':cas,'smiles':smiles,'notes':notes,'blenders':blenders}

def get_onclick_url(result):
    matcher = re.compile(r'openMainWindow\(\'(.*)\'\).*')
    groups =  matcher.match(result['onclick']).groups()
    return groups[0]

def get_all_basedomains():
    url = "http://www.thegoodscentscompany.com/rawmatex-a.html"
    soup = get_soup(url)
    results = soup.find_all(attrs={'href':re.compile(r"http://www\.thegoodscentscompany\.com/rawmatex.*")})
    return [res['href'] for res in results]

all_material_urls = []

for basedomain in get_all_basedomains():
    soup = get_soup(basedomain)
    results = soup.find_all(attrs={'onclick':re.compile(r"openMainWindow*")})
    all_material_urls.extend(results)

print(f'Found {len(all_material_urls)} to crawl.')

for res in tqdm.tqdm(all_material_urls):
    # So as to not kill the website
    time.sleep(.1)
    url = get_onclick_url(res)
    try:
        data = get_data(url)
        with open(f'data/{data["cas"]}.json', 'w') as f:
            json.dump(data, f)
    except AttributeError:
        pass
