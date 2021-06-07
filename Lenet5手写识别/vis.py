
def get_picture(pic_name, transform):
    img = skimage.io.imread(pic_name)
    img = skimage.transform.resize(img, (224, 224))
    img = np.asarray(img, dtype=np.float32)
    return transform(img)
 
 
def make_dirs(path):
    if os.path.exists(path) is False:
        os.makedirs(path)
 
 
def get_feature():
    pic_dir = './input_images/1.jpg' #往网络里输入一张图片
    transform = transforms.ToTensor()
    img = get_picture(pic_dir, transform)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # 插入维度
    img = img.unsqueeze(0)
 
    img = img.to(device)
 
    net = torch.load('./models/1_70/19.pth')
    net.to(device)
    exact_list = None
    # exact_list = ['conv1_block',""]
    dst = './features' #保存的路径
    therd_size = 256 #有些图太小，会放大到这个尺寸
 
    myexactor = FeatureExtractor(net, exact_list)
    outs = myexactor(img)
    for k, v in outs.items():
        features = v[0]
        iter_range = features.shape[0]
        for i in range(iter_range):
            # plt.imshow(x[0].data.numpy()[0,i,:,:],cmap='jet')
            if 'fc' in k:
                continue
 
            feature = features.data.cpu().numpy()
            feature_img = feature[i, :, :]
            feature_img = np.asarray(feature_img * 255, dtype=np.uint8)
 
            dst_path = os.path.join(dst, k)
 
            make_dirs(dst_path)
            feature_img = cv2.applyColorMap(feature_img, cv2.COLORMAP_JET)
            if feature_img.shape[0] < therd_size:
                tmp_file = os.path.join(dst_path, str(i) + '_' + str(therd_size) + '.png')
                tmp_img = feature_img.copy()
                tmp_img = cv2.resize(tmp_img, (therd_size, therd_size), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(tmp_file, tmp_img)
 
            dst_file = os.path.join(dst_path, str(i) + '.png')
            cv2.imwrite(dst_file, feature_img)
 
 
if __name__ == '__main__':
    get_feature()
