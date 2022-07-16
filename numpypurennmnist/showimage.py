import matplotlib.pyplot as plt

#images_2_show = []
#titles_2_show = []
#for i in range(0, 10):
#    r = random.randint(1, 60000)
#    images_2_show.append(x_train[r])
#    titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r]))    

#for i in range(0, 5):
#    r = random.randint(1, 10000)
#    images_2_show.append(x_test[r])        
#    titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))    

def show_images(images,title_texts,fileName):
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(30,20))
    index = 1    
    for x in zip(images, title_texts):        
        image = x[0]        
        title_text = x[1]
        plt.subplot(rows, cols, index)        
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize = 15);        
        index += 1

    plt.savefig(fileName + '.png')



