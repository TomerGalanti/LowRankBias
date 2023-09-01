from conf import global_settings
import csv
import os

def find_xid_pid():    
    xids = [int(i) for i in os.listdir('results')]
    xids.sort()

    xid = str(xids[-1])

    pids = [i for i in os.listdir(os.path.join('results', xid)) if os.path.isdir(i)]
    if len(pids) == 0:
        pid = '0'
    else:
        pid = int(pids[-1]) + 1

    return xid, pid


def log_settings():
    xid, pid = find_xid_pid()
    settings = vars(global_settings)
    exclude = ['os', 'datetime', 'math']
    settings = {key: val for key, val in settings.items() if ('__' not in key and key not in exclude)}    
    settings['xid'] = xid
    settings['pid'] = pid

    # read in previous experiments
    params = []
    experiments = []
    if(os.path.isfile('experiments.csv')):
        with open('experiments.csv', newline='', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            params = reader.fieldnames
            experiments = []
            for row in reader:
                experiments.append(row)
                
    # add new params if necessary
    for param in settings.keys():
        if param not in params:
            params.append(param)

    # write new experiment to csv
    with open('experiments.csv', 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=params)

        writer.writeheader()
        for experiment in experiments:
            writer.writerow(experiment)

        writer.writerow(settings)
    
if __name__ == '__main__':
    log_settings()
