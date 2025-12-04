import os

import requests
import json
save_folder = "downloads"
os.makedirs(save_folder, exist_ok=True)

def fetch_gujrera_projects(district="Ahmedabad", project_type="Residential"):

    url = f"https://gujrera.gujarat.gov.in/dashboard/get-district-wise-projectlist/0/0/all/{district}/{project_type}"

    headers = {
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive',
        'Referer': 'https://gujrera.gujarat.gov.in/',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36',
        'sec-ch-ua': '"Chromium";v="142", "Google Chrome";v="142", "Not_A Brand";v="99"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        # 'Cookie': '_ga=GA1.3.1927037471.1757590963',
    }


    response = requests.get(url,headers=headers,verify=False)
    if response.status_code != 200:
        print(f"Failed to fetch data. Status Code: {response.status_code}")
        return
    try:
        data = response.json()
    except json.JSONDecodeError:
        print("Failed to decode JSON response.")
        return
    results = []
    projects = data.get('data', [])
    for project in projects:
        projectRegId = project.get('projectRegId')
        basic_details  = {
            "projectName": project.get('projectName'),
            "project_status": project.get('project_status'),
            "projectType": project.get('projectType'),
            "districtName": project.get('districtName'),
            "districtType": project.get('districtType'),
            "project_address": project.get('project_address'),
            "projectRegId": project.get('projectRegId'),
            "regNo": project.get('regNo'),
            "project_ack_no": project.get('project_ack_no'),
            "startDate": project.get('startDate'),
            "endDate": project.get('endDate'),
            "approvedOn": project.get('approvedOn'),
            "projOrgFDate": project.get('projOrgFDate'),
            "extDate": project.get('extDate'),
            "disposed_date": project.get('disposed_date'),
            "hardcopysubmisspmtr_email_idionDate": project.get('hardcopysubmissionDate'),
            "payment_status": project.get('payment_status'),
            "payment_token": project.get('payment_token'),
            "project_cost": project.get('project_cost'),
            "projectCost": project.get('projectCost'),
            "total_est_cost_of_proj": project.get('total_est_cost_of_proj'),
            "regFee": project.get('regFee'),
            "wfoid": project.get('wfoid')
        }
        headers = {
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive',
            'Referer': 'https://gujrera.gujarat.gov.in/',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36',
            'sec-ch-ua': '"Chromium";v="142", "Google Chrome";v="142", "Not_A Brand";v="99"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            # 'Cookie': '_ga=GA1.3.1927037471.1757590963',
        }

        response1 = requests.get(f'https://gujrera.gujarat.gov.in/project_reg/public/getproject-details/{projectRegId}',  headers=headers)
        if response1.status_code != 200:
            print(f"Failed to fetch data. Status Code: {response1.status_code}")
            return
        try:
            data1 = response1.json()
        except json.JSONDecodeError:
            print("Failed to decode JSON response.")
            return
        basic_details.update({
            "ProjectLandArea": data1.get('data',{}).get('projectDetail',{}).get('totAreaOfLand',{}),
            "TotalOpenArea": data1.get('data',{}).get('projectDetail',{}).get('totOpenArea',{}),
            "TotalCoveredArea": data1.get('data',{}).get('projectDetail',{}).get('totCoverdArea',{}),
            "PlanPassingAuthority": data1.get('data',{}).get('projectDetail',{}).get('approvingAuthority',{}),
            "Taluka": data1.get('data',{}).get('projectDetail',{}).get('subDistName',{}),
            "District": data1.get('data',{}).get('projectDetail',{}).get('distName',{}),
            "State": data1.get('data',{}).get('projectDetail',{}).get('stateName',{}),
            "pinCode": data1.get('data',{}).get('projectDetail',{}).get('pinCode',{}),
            "moje": data1.get('data',{}).get('projectDetail',{}).get('moje',{}),
            "costOfLand":  data1.get('data',{}).get('projectDetail',{}).get('costOfLand',{}),
            "estimatedCost": data1.get('data',{}).get('projectDetail',{}).get('estimatedCost',{}),
            "totalProjectCost":  data1.get('data',{}).get('projectDetail',{}).get('totalProjectCost',{}),
            "AboutProperty": data1.get('data', {}).get('projectDetail', {}).get('projectDesc', {}),
            "tPNo":  data1.get('data',{}).get('projectDetail',{}).get('tPNo',{}),
            "plotNo":  data1.get('data',{}).get('projectDetail',{}).get('tPNo',{}),
            "tPName":  data1.get('data',{}).get('projectDetail',{}).get('tPName',{}),
        })
        englist =  data1.get('data', {}).get('englist', [])
        acrchlist = data1.get('data',{}).get('acrchlist',[])
        Contrators = data1.get('data',{}).get('contr',[])
        Agent = data1.get('data',{}).get('agentlist',[])

        headers = {
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive',
            'Referer': 'https://gujrera.gujarat.gov.in/',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36',
            'sec-ch-ua': '"Chromium";v="142", "Google Chrome";v="142", "Not_A Brand";v="99"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',

        }

        response5 = requests.get(f'https://gujrera.gujarat.gov.in/project_reg/public/alldatabyprojectid/{projectRegId}',headers=headers)
        if response5.status_code != 200:
            print(f"Failed to fetch data. Status Code: {response5.status_code}")
            return
        try:
            data5 = response5.json()
        except json.JSONDecodeError:
            print("Failed to decode JSON response.")
            return

        promoterId = data5.get('data',{}).get('promoterId',{})
        headers = {
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive',
            'Referer': 'https://gujrera.gujarat.gov.in/',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36',
            'sec-ch-ua': '"Chromium";v="142", "Google Chrome";v="142", "Not_A Brand";v="99"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            # 'Cookie': '_ga=GA1.3.1927037471.1757590963',
        }

        response3 = requests.get(f'https://gujrera.gujarat.gov.in/user_reg/promoter/promoter{promoterId}', headers=headers)
        if response3.status_code != 200:
            print(f"Failed to fetch data. Status Code: {response3.status_code}")
            return
        try:
            data3 = response3.json()
        except json.JSONDecodeError:
            print("Failed to decode JSON response.")
            return

        assosiate = data3.get('assosiateList',[])
        authorizedSignatory = data3.get('authorizedSignatoryList',[])
        Promoters = {
            'id' : data3.get('id',{}),
            'Promoter Name' : data3.get('promoterName',{}),
            'Promoter Type' : data3.get('promoterType',{}),
            'Contact' : data3.get('mobileNo',{}),
            'Email Id':data3.get('emailId',{}),
            'faxNo': data3.get('faxNo', {}),
            'panNo': data3.get('panNo', {}),
            'adharNo': data3.get('adharNo', {}),
            'status': data3.get('status', {}),
            'companyRegistrationNumber': data3.get('companyRegistrationNumber',{}),
            'companyRegCertificateDocId': data3.get('companyRegCertificateDocId', {}),
            'entities_developerGroupName': data3.get('entities_developerGroupName', {}),
            'Total no. Of Years Of Work Experience Of Group Entity In Gujarat': data3.get('entities_experienceGroupEntity',{}),
            'Total no. of Ongoing Projects By Group Entity': data3.get('entities_totalProjects', {}),
            'Total Area Constructed Till Date By Group Entity For Completed Projects(Sq Mtrs)': data3.get('entities_areaConstructed', {}),

        }
        headers = {
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive',
            'Referer': 'https://gujrera.gujarat.gov.in/',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36',
            'sec-ch-ua': '"Chromium";v="142", "Google Chrome";v="142", "Not_A Brand";v="99"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'Cookie': '_ga=GA1.3.1927037471.1757590963',
        }

        response5 = requests.get(f'https://gujrera.gujarat.gov.in/form_five/get-formfive-records-till-date/{projectRegId}', headers=headers)
        if response5.status_code != 200:
            print(f"Failed to fetch data. Status Code: {response5.status_code}")
            return
        try:
            data5 = response5.json()
        except json.JSONDecodeError:
            print("Failed to decode JSON response.")
            return
        for i in data5:
            try:
                pdfUid = i['pdfUid']
            except:
                pdfUid = ''
            if not pdfUid:
                print("No pdfUid found for record.")
                continue
            pdf_url = f'https://gujrera.gujarat.gov.in/vdms/download/{pdfUid}'
            print(f"Downloading: {pdf_url}")

            # Download the PDF
            pdf_response = requests.get(pdf_url, headers=headers)

            if pdf_response.status_code == 200:
                filename = os.path.join(save_folder, f"{pdfUid}.pdf")
                with open(filename, "wb") as f:
                    f.write(pdf_response.content)

                print(f"PDF saved as {filename}")
            else:
                print(f"Failed to download PDF for {pdfUid}. Status Code: {pdf_response.status_code}")

        item = {
            "ProjectDetails": basic_details,
            "Engineer": englist,
            "Architects" :acrchlist,
            "Contrators":Contrators,
            "Agent":Agent,            "assosiate":assosiate,
            "Signatory Details":authorizedSignatory,
            "Promoter Details":Promoters,
            "pdf_url":pdf_url,
        }
        results.append(item)
        print(results)
        with open('gujrera_Residential_ahemadabad_1.json','w') as file:
            file.write(json.dumps(results,indent=2))

if __name__ == "__main__":
    print("Fetching Gujarat RERA projects for Ahmedabad district...")
    for project in fetch_gujrera_projects("Ahmedabad", "Residential"):
        print(json.dumps(project, indent=2))
        print("-" * 80)
