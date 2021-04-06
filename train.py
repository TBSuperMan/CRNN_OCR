#coding:utf-8
from __future__ import print_function
from __future__ import division

import sys
#sys.path.extend(['/home/yhy/.conda/envs/yhy/lib/python3.8/site-packages'])

import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from warpctc_pytorch import CTCLoss
import os
import utils
import dataset
import os.path as osp
import models.crnn as crnn
from load_save import *
# from torch.utils.tensorboard import SummaryWriter
# import torchsnooper

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
parser = argparse.ArgumentParser()
# parser.add_argument('--trainRoot', type=str, default='D:/Document/DataSet/DataSet/Chinese_data/MTWI_train_filter', help='path to dataset')
# parser.add_argument('--valRoot', type=str, default='D:/Document/DataSet/DataSet/Chinese_data/MTWI_test_filter', help='path to dataset')
parser.add_argument('--trainRoot', type=str, default='/home/gmn/datasets/MTWI_train_filter', help='path to dataset')
parser.add_argument('--valRoot', type=str, default='/home/gmn/datasets/MTWI_test_filter', help='path to dataset')
# parser.add_argument('--trainRoot', type=str, default='/home/gmn/datasets/NIPS2014', help='path to dataset')
# parser.add_argument('--valRoot', type=str, default='/home/gmn/datasets/IIIT5K_3000', help='path to dataset')

parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--nepoch', type=int, default=500, help='number of epochs to train for')
# TODO(meijieru): epoch -> iter
parser.add_argument('--cuda',default=1, action='store_true', help='enables cuda')
parser.add_argument('--resume', type=str, default="/home/gmn/crnn/checkpoint_keep_ratio.pth.tar")
parser.add_argument('--pretrained', default='', help="path to pretrained model (to continue training)")
parser.add_argument('--alphabet', type=str, default='兔樹运捆亏嗓邀息貝捕农釜蔻洒秦赠吗叔骑殼胞遠斤飴议嬰倡宝腔郁孕田蜡馆俪镂限租膠香绎潭舰辩叟驗蛎文黑糖店歷窦蕾莹但败涵姬阶洪厅俊屆徽哥学肥難您薬厢灶媛枕唾窃囊蛛电撼权击跳顏状固爽嘉途之菁给漾氢潤杰媄淑理团迅鳄肘檫鼠送弟钒蔬奈货懂晞筷带便额塞汗缀裹語澈啫颂笔到璋炔袆驿生何李妍蛋腩构國默淄帯林宿闽泻勐解驻杭蒲科嫁烫膛艾怕闰传类濃属羞枣胸扭類糊痱虾苗圣顶驱吕車披奕蘭演乘涔队試彤藿延捡裸韵狗椰偌纺棕非镍谓凤趣群顽鼻草刃峦痴兄焙種举畅束侈栓掘碗界辰亡猩签例珊榉惹超侦牧蕒云酬東枸眉托弗钜親蒙闷砂间略湛废土弛阙布旗董鎖熟句至朝程必克氨灣女洁寺帆莉紫孜休桶红桌导宽鹃江本梯蕨近矩疲悦芦碱様车芍吸副屌搐们形宫镌舜湖造迟樊荀饵势墅楓麯辣鲩肆尚否燙鲟须熔喬玖陽澜啖浸凶蝎椹柠瞌判矫楠粘精迫她替聆镰蜀黃廉极干归沂沥陌糕喝探名啶仕里仪牙嗖飾件渭候2一购饲拼虱满占浒补褥优莫咭乳留瑶拽径增歌氟冀艺傳調敦無気点踏凈缨止鹤罩纱档滤呐腐鸦蘋骊總州强據斐從图扑墙嘴那逹鉴守抄久丸铁芪設阱树帜碑荐肃缇烊極罘尹琢刺底叛效營似靜尺柄微牢埋愣较寓厂躁添目馴并滕铛希彻郊沁翠伸杯结粉乓捏钢椎晕琬咨兜棵赣內茵乱淋寶随馅獸赞疵懿髅溶装牵难钞悟借颗鸾伪镊巣舱臣惯淘投擎巅柿未佑灸挂荣坚岩荆氧柑喽该路阴登偿筑戊晔京侬財憨矿證黔扩镭藏蛤绘民神圆識身铎咱梅衡专为晟需届尤纤骄煖甩秧陶學历籽泾斑鹦餐圳宋失塗茯欲跪梓猪晒毁殁锻促備写寸玛钩付橄論舔竿琐闭珀尬蚕攝疏衛锂専哆宾面搽亜镶庸削炸锣筝纯祥基响澡能慕咿颐拎係帥蜂夕寇碎綱城猴纷率热哪机墨酥枼蒜紐妙昭業青掟殿柴封畔舟滴盛根湾知沉漩毯癫润因卷国魔眠婧慢尘疾币煙员綫曲叱奉祖狐谧汪汉將撬缕呕汝票做库狄际衆聘四陷坊藤疯己楚涤谨卢糠旅集谜吨筛博丑舞扫訂皎制晖被虎穗边整律純棘鍋説鲨话想层断森雁咏忏焗該厌房薇骆脱函倒绞范珺欧哮旋漸耍荞零策它销刘鮮寒洞禦芜舖沌翻藕闯豹陰沽方搬忐囍纠涡瑕剩损阔钊褐冕恰処帛缝髪侣劫度痧歡跃吐刷绕對豫榄验婕嚴抑遮呆岂亮存音碼垂瀚陈栽伐缔桉缺摩筠馒崎嬉星许點参涩脊妹扁卤鲁資选哒荟截餠稍鲍顿灭辧自兹鄉令场缸燥耕伤缘璇仞麻勤妻顺玷铆龙级松驴择细赌兽睫歇橋昨缈醫逛建端究遗翡穷柒聯畸伴頭輝统轻銷画楼俺凹蓓町笛绣俄肽烘壕檬垛严抽洛瑟中背美各膝乃伙二拖哲划薪牛魁桩纂穴校纇霞哎铸脾赏秉菲剪亚夷姓诞剔甜蕲残诱皇g莲渐薯所骐禁朕锈翎仿奶苍愛飮陇录巍溺團袜配换節尽营符仔哇爾苯功个雀芷负考假缆淚数混锁孩佬弃裘鸿計圾箐域高皆佐妥坨捍艶費谭饥枝逝况贵匙锯偉棺拨 查悲呗瞎翘岑摘吵鹏繹蝌楂孢喫靖浚終锌入萌比這侵旁雪栈硬姜惠映萄0姆蛭灵周凌連皙崋灰钳扒珠识肚调现盎竖淨逍出转篮壳袍气挖唇央隅撻费污娠浆献浅養段礼曝丨且洗唑滋诠現壹窄叮凭轨婆晗喷绷征仅篷權給夢泡颊簧筒貂吧倍麥細既璃很顯廿赋藥倾鐡返曼困绝兒汽执颠搓悅谐贱員溃妇嗮鑫喔渝碁織鲷茉企挠晴狭茜樂汞扔砸幂棋纶皴招站狂初媳誓鍾桐泷壁毫序韦貔武找滢帖棠葡蔽剥溪泵狱桂他緣紧酯花余罐将座陪西芮皖咤拔卉荷办阵聊馨創炽触批屉受莞早岗瑾核羙蒻凱曹缎催障聚戏的孙產党嬷璨葫错聿帚損虽饪篇龈挤甄闸宠条裙哦題哚维窥盏袋扰祛忠户绿枯臀陳产杆衰像滙赵俱蹭北碩待燈豺定膳军铃潮仟内枭達義堵惟晶磺益璀呢在擦只佛蓝午糯曬屋缥降瑙笋羔乒換恒冏邦濠稀涌恶煌叭炼郑發流桧订总烟衫偏孝芃忙芽贴炀4步.掸媚續玮合鞋净磕揭帘竹力咬琥芳涇関柱爲全号佰氯滑庄婊告观療业泛惧稳葉铜架勇斯郎歸庞蒄蒂模殖戴重贅妝隐晃蚌贰臂持劲報樣摊玄收几银钠販盗坏蜜来频湓糟巨粕钱郵勺供榛再绢钰烦琯距柔跌就啸浦尔韬争味旺拉左闻戰认骷院氛艷均阀蛊拷黄双瓷誠範疫防徕智賠绚宗絶埠苪不蛀盔鹅储泞撞嘧家老男顔努喧箔禮什昊块蛮秋夾沟起喂裕行愁橙纳潇咽设壊连舆箭塔為跨艇沫锋救石呵后屁胧隔税绗廣裁讀玥姐或姚危邂棚痰泣置寄塑足蚂邮經窖梦菊光锤免芋破壶研八孔丫拯晳涯縣舌靈睁以积丝褪熠师拿狠崇洲猫娥室逻啪呛肤赚展膜丽蕊睿烤玻别永捂徹纽璐剛叶桥鵬遇噜滩盖樟鹄醛仇泽蚁坛弓弄橘饅采骨鉅兴饿辨纸拳傲绳瓮烈答還母肖预折哺會枪上惬綁裝递奋朋厉彈当鲜疑鳞昕酷杂算载甸鏜崖脖贈逢罚泌赁斋巫膏鬃見澎摔檐株俫圖烁毡逐萝亞薦巾漳棉么仙宏纪氣活瑰齐嚼斛蠕伏硝笼芒丹咪黒湘糙趋酐烯联钥循磅父子普啰辜萧赔頰減酱嘿实窘娜谦排婦獻務格製晾動雨鴨空浓倪逃芹雕補聖滨阎怎街扣熱尊俗崬经辫獭鐘助厦尼镜冈鼙貴伽括详師创弥谣遂淼声竞跑抛族磁十褚灯箸鸡岳楷賀媽火饼绩寡士亦暴氏銑楸信散麺枚酸昱政苇挑我貨谱坤惜熨式包象帶关壽屏咆贩羊幽吊甫契罪匠帮肯燃祸冒见品挫叫杏窗抱薏突奖偶穆日欣斗笨接阿戈劵福刊已鎏縫焼愈心蟀言尸搜担黏7冠熊马烽儥看你众醉伦岁枞础刨疙崩夹痘评挚池扮独器鸳钽摈即寢暇尿读呀豆口冬吖魅螂嘻康钛券陆透塘嗣异勿培挎套豚诺摇相五龚召摆沾放炮撒酮谋彦腻命舍凉煸逺系暂腌乎表抵泰辞铣临姒练辅钮寫脚乔朵凄羽冰岭吉貼裆呼陕瓶耀婷強暑奇觅徒泼腰汤责傢同娇乡盆谷呂揉風每曦多还珏兑停喰櫥激环臻潘彼太讯迷忑瞳附唢堰瞬秒運御翁殊獨砧峡链好赖霉灾拐案蒟龟澤沪蝇嫩蛇橱漓哑剁则取裱于亲涮洽桃援劳庆渔东霜煎詢筆改絮代時峰臨蔓朔苔儱幻瑜诸恐侠续简据忍熙晚凛疝提脂夺卜仲惑禹料犸愉计鱼傻葱践恩耦胺描奥虫锵尐祺娘汐馬凍粹貳捉长职朱黯榨愿快禧琦桔吴挥皓最疼勒盤荧虚芝專泸私靑熏禅有榻瑄线媒照班匀進舊瘦約茧百君贼阜寕毒贝厘夏绑疣謢妖证恋鼓锰软雄诗灿肝庭荡廷柚地盼蕉与勁舵霏碟夲鳍皮穿閟友劣栏膀漢撃苏誉唐肺撇骤蒸沃輔融慎型浙歪米钦良胜浪繁酶嘟动拟万沏秀雍锅逼今冻野开值瘊称涛長吹雷騷钼弯辆幸航船琳瀛竟司犹攻狒颜份古委猜烷敌月景呈煊悠侍低手决幼这蹲事吻懒常始刮志憋讲译欢夫儿可躺環崔張纲毂宅編疹邯斜客眸课意粵卿部拆緑瓦麓茸質共麒刀史蠔锗茬谁廠广鴻6沈菜萍兰使对彬缅涨鷄爹紅色财鸢漫啡医胆敛飞凯組丁碘麝锶清籍厕珑疆膚染利喉南瑪痣奢是蜕秤岛檀哟棒轧芸衍針拍晋錯尖糜袭妈踩禪源框准華复汇錶歓伯醬丰魂褂谊厚俩署获视胀芊恭辈赐碰啥龄軍其冉妳六巧态娣处1诵缤如靴犀與砍轴罄飼囡壮渗梵挺消诉伟湃主兆络埃真曙性下振呦删孤烹护熬斧肴酉鸯橡咯菩握郝淳硼萨凸烨钟缦磐槛特九灼注餅桨輪磨悄驾臭茨溢织腱骏抚过材恢把屑逊棍孚旨版冽鳕灌少而鬼嗔拾標头鱸脐杞派典惫审习腊物疗玫享游勢诚应钧镇啦概妃茂遍胚囤市编闪芬互咻健抗圭窕蓄幅览勃直荔申售般管娅臥虐药外饱宁蟆菱鄂匹拌疤惆巴台乾若通蝶压旦球琮訊椅从炭先雜弦赶蝴卸瘩藓闹喱账丛淇靓艳嵩螨甲鞭羚浊然脸引翰官退启暖岚肢索赢粤浏洋酒泪册歲听佩p减要亭爬规貅亨眞焦烏峻笑渠彪酊億盈昔赴检榕桁潍贤瞅葆镀幚禾释妩脓影惊两豪着迪煞扳啃彭坐痕砭楦港釉向戌碧裳济确移射枰骗杠肌咕弱针炉蹈謹前筐寂旭燊詩勝両悔砖磊剑脆慈溫够揚氓期坦藻浑裂陀扶泊赫翌粒齊夜字毕椒板剂辛钣皲顾术娴张羡盾雙籁翼股蟑紹邪床爸嚣半記琪春焊刹躲庚拜邹列堡幹粗梧血杜输铺莎革述價杨绅差寞宪實弧箱缩显又靠婉哈卫驭痒吲興釀j交鲸此七炣凳離昇膨啾绪摸祝螺险瑚世炜速萬横阅新抠求宇堆帽溜胃倆皂焰賞妆雑海钙盒俏芥猎氮珂虑玲堂窍蒋适怒応魄量浩窈蟋予笈宛饭唱栎吾術坠魏跟波械泥珞淀寝保迈鲤浴僧渴槽甬島孟蜗焖潜袖致踪缴酿叠梢怡焕鸭芙珍陵罂锥绵和忘蘆肿沱皺胡沐拥凰坡朦植澄滚问敏巩思頓鉄雯剃漱贸寻姨瞿训违亿聪硒义賣白测雲综蓉河絲乌锦機娟炫嗽素急磷炒酪沿覆液廊榜辟洙宣越兵殺叉腮语稿瓜木扇鞘腿猛雇硅霾钨技铡韩拭童網紀肠婚无咧璎昵尷裔兩梳搭纬忄觉辉卓咖骚果翔滔搁肉址測簡闺韓切館钉劑进撕赤作咔盜咚鬚漏迁胤浮章來醇犯哗榴諾腋瓣颖分祁黎剧泄盟褲警症诊种粮德載颁深丘忧正绍寨威薰艦巡颈刚毅柜梭窝容绸産錄锆栖凡逅爵忆痔纹盐绰馈桑含抖蝠汕單质震咀勾贮紊亩逆虹緯閃耻栅浣察刑購询倚卍鋒让报軽摄8熥脉服押醋芭慧款宙拓约奴霓变肩裤体伞挡沙涧铬蕴偷硫离辑煲静感筏组网井华碳舒山稚食衣胰豇琼鳥哄睛逸除璟苹苦蚀翅僑记數衬龋操尝风朗走铝津大烙园厨敷郸回潔乖鹿洱鍵炖刻菇完法哩施噬及绒晓朴韧励阳邢标枫竭漯鹰麗痛谅漆隋饮次试翟拱瘤門玺公縮鸣没曉奔眼輕湿靡者支伺盅晰撸屎肾买社帕秘肪葛县耐壬腺馏具軒猬凖貌片绻铅督养析霆宜谢衔冤温钓追糝箍打才承迎缠柏袁鋼番汁萤琅喵了仓間恊饶捷娃露掌忽鐵邵擒居倩巢偽橹慵電商弹闲粧冷资爷狸芯位決烧楞悵仑抢対凿咒啤韶霖沸痞卧挞年协琴優娱席噪護敢甘睡时阁指说易稔仝婴金垃卵右往焱哭麦围伍樱卖泉苓缓筋鹉爱辦区加胶犬閩腹莊履轮搏卡冯会欺淮黛领熳天鎢鬓疮撑蔗立育绨忌千層饰隆奂旧帝贺依著彩患癣薈平蠶蛳锐毛菀怖槟銀酚死緖化幕密累第脑際訣虞锡蜘秩昌菠順喘继溅原掉蟹葵關捞燕请蛲盲兼鯽伱侧牡辽谈槐冲歆英铠钻儒烂隙啄由揽驰炎府键鼎藝割菌丙人备怀辐誌嗪戒伊情唤轩蘑侨槍乐播僮授明蓬吃茶豌瑞战误馥鹭苑皱也挪莓节介纵蜱掺首蔡酵唛书鯉狮聲鲶蚊铂姗廓遵號尾杉茗乙雾獅耳丶油更輼冶敬词宸弘搅王罗胖境骼去煮雅暗嘘貿勋紗避浇渍萃祈充悉體寿害却醒妊憬噢逗薄哨咳角念過嗨叻園茄蚤病曾纫省铰工啊胎龍串稠发汀控末粽鋪远夯遥丢缚渡推彰钾岸拒码腾恼維狀铭婭富莱畜挣鸥曜道捶稻溉都涂搞漂终扎脏僵徐攸村烛氙牌飄住梁級偕爆笙某傅修妞寳升贡髮楊碌哼门仰傘嗑绮怪水悍反粪价阻玉戳5杀粥望局睦轿泳抓示鸟赛帐藍拢妮扬贯垢腕携悬孵题故侯喜場魚按元季荒教煜雌萱嵌奏访昆善凝柯豐碍圈项任等监填烩镁佳安斌耗梨三得插达吓宴導印榔绽恨蛰捧楽寧扯抹敲鸽迹喇郭姑霸蛔姿澳矶屹治苡铲晨唯崽川邻驶样渣娶迭適用狼论萎另舶仁開驼短务页硕缉攀单帅柳蹄傷垫成書盘淡丈飘小落些蚪齿爪玩飛')
# parser.add_argument('--alphabet',type=str,default="0123456789abcdefghijklmnopqrstuvwxyz")
parser.add_argument('--expr_dir', default='expr', help='Where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=1, help='Interval to be displayed')
parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display when test')
parser.add_argument('--valInterval', type=int, default=10, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=100, help='Interval to be displayed')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate for Critic, not used by adadealta')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--adam',  default=0, action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--adadelta', default=0, action='store_true', help='Whether to use adadelta (default is rmsprop)')
parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
parser.add_argument('--manualSeed', type=int, default=1234, help='reproduce experiemnt')
parser.add_argument('--random_sample', action='store_true', help='whether to sample the dataset with random sampler')
opt = parser.parse_args()

#日志
logger=utils.get_logger(osp.join("logs", "CRNN_keep_ratio.logs"))
# writer = SummaryWriter(comment="crnn_keep_ratio")

if not os.path.exists(opt.expr_dir):
    os.makedirs(opt.expr_dir)

random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

acc_max=0.0

if torch.cuda.is_available() and not opt.cuda:
    logger.info("WARNING: You have a CUDA device, so you should probably run with --cuda")

#获取训练数据集
train_dataset = dataset.lmdbDataset(root=opt.trainRoot)
assert train_dataset
#看是否是随机采样
if not opt.random_sample:
    sampler = dataset.randomSequentialSampler(train_dataset, opt.batchSize)
else:
    sampler = None
#从数据集中加载训练数据
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batchSize,
    num_workers=int(opt.workers),drop_last=True,shuffle=False,
    collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW,keep_ratio=True))

#从数据集中加载测试数据
test_dataset = dataset.lmdbDataset(root=opt.valRoot)

logger.info("train:%d  test:%d" % (len(train_dataset) ,len(test_dataset)))
nclass = len(opt.alphabet) + 1
nc = 3
#定义转换规则
converter = utils.strLabelConverter(opt.alphabet)
criterion = CTCLoss()


# custom weights initialization called on crnn 参数初始化
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


crnn = crnn.CRNN(opt.imgH, nc, nclass, opt.nh)

# setup optimizer
if opt.adam:
    optimizer = optim.Adam(crnn.parameters(), lr=opt.lr,
                           betas=(opt.beta1, 0.999))
    logger.info("--------------------------------------adam--------------------------------------")
elif opt.adadelta:
    optimizer = optim.Adadelta(crnn.parameters())
    logger.info("--------------------------------------adadelta--------------------------------------")
else:
    optimizer = optim.RMSprop(crnn.parameters(), lr=opt.lr)
    logger.info("--------------------------------------RMSprop--------------------------------------")
# logger.info("--------------------------------------adam--------------------------------------")
#训练模型的验证部分

if opt.cuda:
    crnn.cuda()
    crnn = torch.nn.DataParallel(crnn)
    logger.info('cuda is available')
    criterion = criterion.cuda()

if os.path.isfile(opt.resume):
    checkpoint = load_checkpoint(opt.resume)
    crnn.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optim_state_dict'])
    start_epoch = checkpoint['epoch']
    acc_max = checkpoint['best_res']
    logger.info("=> Start iters {}  best res {:.1%}".format(start_epoch, acc_max))
else:
    # 初始化权重
    crnn.apply(weights_init)

image = torch.FloatTensor(opt.batchSize, 3, opt.imgH, opt.imgH)
text = torch.IntTensor(opt.batchSize * 5)
length = torch.IntTensor(opt.batchSize)



image = Variable(image)
text = Variable(text)
length = Variable(length)

# loss averager
loss_avg = utils.averager()


def val(net, test_db, criterion, max_iter=100):
    logger.info('Start val')

    for p in crnn.parameters():
        p.requires_grad = False

    net.eval()
    data_loader = torch.utils.data.DataLoader(
        test_db, drop_last=True,shuffle=True, batch_size=opt.batchSize, num_workers=int(opt.workers),collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW,keep_ratio=True))
    val_iter = iter(data_loader)

    i = 0
    n_correct = 0
    loss_avg1 = utils.averager()

    max_iter = min(max_iter, len(data_loader))
    for i in range(max_iter):
        data = next(val_iter)
        i += 1
        cpu_images, cpu_labels,lens = data
        batch_size = cpu_images.size(0)
        utils.loadData(image,cpu_images)
        # logger.info(cpu_labels)
        t, l = converter.encode(cpu_labels)
        t = torch.tensor(t)
        l = torch.tensor(l)
        utils.loadData(text, t)
        utils.loadData(length, l)

        preds = crnn(image) #[26,batch,nclass]    #[batch,26,nclass]
        # print("-----------------------------")
        # print(preds.shape) #[104,16,3783]
        preds = preds.contiguous().permute(1,0,2)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size)) #[batch*26]
        # logger.info(batch_size) #64
        loss = criterion(preds, text, preds_size, length) / batch_size
        # logger.info("loss.shape: %s" % (loss.shape))
        loss_avg1.add(loss)
        _, preds = preds.max(2)
        # preds = preds.squeeze(2)
        preds = preds.permute(1,0).contiguous().view(-1) #[batch*26]
        # logger.info("preds_size",preds_size.shape)

        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        for pred, target in zip(sim_preds, cpu_labels):
            if pred == target.lower():
                n_correct += 1
    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:opt.n_test_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_labels):
        logger.info('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))
    accuracy = n_correct / float(max_iter * opt.batchSize)
    logger.info('Test loss: %f, accuray: %f' % (loss_avg1.val(), accuracy))
    loss_avg1.reset()
    return accuracy

# @torchsnooper.snoop()
def trainBatch(net, criterion, optimizer):
    data = next(train_iter)
    cpu_images, cpu_labels,lens = data
    batch_size = cpu_images.size(0)
    # logger.warning(type(images))
    # logger.warning(images.shape)
    # logger.warning(cpu_labels)
    utils.loadData(image,cpu_images)
    t, l = converter.encode(cpu_labels)
    t = torch.tensor(t)
    l = torch.tensor(l)
    utils.loadData(text, t)
    utils.loadData(length, l)
    # logger.warning(t)
    # logger.warning(lens)

    # logger.info(image.shape) #[64, 3, 32, 100]
    preds = crnn(image)
    preds = preds.contiguous().permute(1, 0, 2)
    # logger.info(preds.shape) #[64, 26, 3815]
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    loss = criterion(preds, text, preds_size, length) / batch_size
    crnn.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

start_epoch=0

for epoch in range(opt.nepoch-start_epoch):
    train_iter = iter(train_loader)

    for k in range(len(train_iter)):
        for p in crnn.parameters():
            p.requires_grad = True
        crnn.train()
        try:
            loss = trainBatch(crnn, criterion, optimizer)
        except StopIteration:
            pass
        # writer.add_scalar('Loss/train', loss, k)
        if k%200 ==0:
            logger.info('epoch:%d train:%d loss:%s\n' % (epoch, k , loss))
        loss_avg.add(loss)
        if k % 1000  == 0:
            acc =val(crnn, test_dataset, criterion)
            # writer.add_scalar('acc/train', acc, k)
            flag=False
            if acc>acc_max:
                acc_max=acc
                flag=True
            # acc_max=max(acc,acc_max)
            save_checkpoint({
                    'state_dict': crnn.state_dict(),
                    'optim_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_res': acc_max,
                }, flag, fpath=osp.join("/home/gmn/crnn", 'checkpoint_keep_ratio.pth.tar'))
            logger.info('[%d/%d][%d/%d] acc_max:%f' %(epoch, opt.nepoch, k, len(train_loader),acc_max))

    if epoch % opt.displayInterval == 0:
        logger.info('[%d/%d][%d/%d] Loss: %f' %(epoch, opt.nepoch, k, len(train_loader), loss_avg.val()))
        loss_avg.reset()
