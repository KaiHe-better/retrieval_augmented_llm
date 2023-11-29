retrieve_doc = [1,2,3,4]


res = "Doucment {}:".format(index).join(i for index, i in enumerate(retrieve_doc))


print(res)