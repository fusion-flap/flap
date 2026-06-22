# -*- coding: utf-8 -*-
"""
Created on Wed May 11 21:22:18 2022

@author: ShendR
"""
import difflib

options=['Image x','Image y']

in_1='image'

best = difflib.get_close_matches(in_1, options)

# if True in [char.isdigit() for char in in_1]:   
#     print("The string contains a number")
#     try:
#         a=eval(in_1)
#         print(a)
#     except:
#         print('Cannot eval input')
# else:
#     print(in_1)
#     print("The string doesn't contain a number")
   
# print(out_1)

variable = {'Time': [3,4], 'Sample':range(0,10,1), 'Unit': 'V', 'inter':[1e2,1e3]}

def check_in(in_put):
    if in_put in variable.keys():
        best_out = in_put
    else:
    
        if True in [char.isdigit() for char in in_put]:   
            try:
                best_out = eval(in_put)
            except:
                print('Cannot eval input')
                best_out = in_put
        else:    
            my_file = open("options_list.txt", "r")  
            data = my_file.read()
            opt_list = data.replace('\n', ',').split(",")
            
            best = difflib.get_close_matches(in_put, opt_list)
            if best == []:
                best_out = in_put
                # print('Invalid input -- Error')
            else:        
                best_out = best[0]
                
                if best_out == 'True':
                    best_out = True
                    
                elif best_out == 'False':
                    best_out = False
                    
                elif (best_out == 'None' or best_out==''):
                    best_out = None
                    
            my_file.close()
            
    return best_out


def check_if_list(in_put):
    if ('[' in in_put and ']' in in_put):
        a = in_put[in_put.find('[')-1 + 1:in_put.find(']')+1]
        b = eval(a)
        in_put = in_put.replace(in_put[in_put.find('[')-1 + 1:in_put.find(']')+1], 'list')
        
    if ',' in in_put:
        if ' ' in in_put:
            in_put = in_put.replace(' ','')
        in_put = in_put.split(',')
        out_put = []
        for i in in_put:
            out_put.append(check_in(i))
        if 'list' in out_put:
            out_put= [b if x=='list' else x for x in out_put]
        # if 'Error' in out_put:
        #     out_put.remove('Error')
        #     print('Invalid input element was removed from list')
    elif in_put == 'list':
        out_put = b
        
    else:
        out_put = check_in(in_put)
        
    return out_put
    

def make_dict(key, value):
    dict_in = {}
    try:
        keys = check_if_list(key)
        values = check_if_list(value)
        if (type(keys) is list and type(values) is list):
            if len(keys) != len(values):
                print('Numer of keys does not mach with number of values!')
            else:
                dict_in = {keys[i]:values[i] for i in range(len(keys))}
        else:
            dict_in = {keys:values}
    except:
        print('Could not make Dictionary!')
        
    return dict_in

input_1 = 'Time'
input_2 = '[2, 3]'


# best_out1 = check_if_list('image x, image y, devicex, [2, 4], 1134.4')
# best_out1 = check_if_list(input_1)
# best_out1 = check_in('image x, image y')
best_out1 = make_dict(input_1, input_2)
# best_out1 = check_in('TEST-1-1')
# dic = make_dict('Log x, Log y, Range', 'True, True, [2e-3,1e2]')
# best_out1 = check_in('inter')
# best_out1 = make_dict('Chop, Defl', '0, 0')

print(best_out1)
    




