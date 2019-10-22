dog_filter_list = ['dog', '🐕', '🐶', '🐩', 'mutt', 'puppy', 'puppies', 'pup', 'pittbull',
                   'bulldog', 'lab', 'labrador', 'pug', 'poodle', 'pooch', 'beagle',
                   'boxer', 'terrier', 'collie', 'border collie', 'corgi', 'chihuahua',
                   'golden retriever', 'retriever', 'german shepherd', 'hound', 'greyhound',
                   'foxhound', 'wolfhound', 'hunting dog', 'rescue dog', 'crossbreed dog',
                   'mixed breed dog', 'sheepdog', 'lap dog', 'guide dog',
                   'watchdog', 'mongrel', 'rottweiler', 'schnauzer', 'dachshund', 'husky',
                   'puggle', 'catahoula', 'yorkie', 'yorkshire terrier']

dog_filter_list_1 = ['akita', 'adopted dog', 'abused dog']

dog_filter_list_2 = ['canine', 'k9', 'bark', 'woof', 'wag', 'собака', '개', 'perro',
                     'chó', '犬', '狗']

phrases_dog_ownership = 'my '+'|my '.join(dog_filter_list)+\
'|I have a '+'|I have a '.join(dog_filter_list)+\
'|I adopted a '+'|I adopted a '.join(dog_filter_list)+\
'|I rescued a '+'|I rescued a '.join(dog_filter_list)+\
'|I have an '+'|I have an '.join(dog_filter_list_1)+\
'|I have two '+'|I have two '.join(dog_filter_list)+\
'|I got a '+'|I got a '.join(dog_filter_list)+\
'|I own a '+'|I own a '.join(dog_filter_list)+\
'|моя собака'+'|у меня есть собака'+'|내 강아지'+'|나는 개를 가지고있다'+\
'|mi perro'+'|tengo un perro'+'|con chó của tôi'+'|tôi có một con chó'+\
'|我的狗'+'|我养了一条狗'+'|うちの犬'+'|私は犬を飼っている'

cat_filter_list = ['cat ', 'cat,', 'cat.', 'cats', '🐈', '🐱', 'tabby', 'pussycat', 'puss ', 'kitty', 'kitties', 
                   'kit ', 'kittycat', 'tomcat', 'kitten', 'rescue cat', 'birman',
                   'house cat', 'mixed breed cat', 'himalayan cat', 'burmese cat', 'siberian cat',
                   'persian cat', 'ragdoll', 'korat',  'siamese', 'maine coon', 'forest cat',
                   'bengal cat', 'shorthair', 'longhair', 'wirehair', 'calico', 'maltese cat',
                   'british blue', 'russian blue', 'manx', 'russian cat', 'sphynx', 'bobtail'] #'cat,', 'cat.', 'cats',

cat_filter_list_1 = ['indoor cat', 'outdoor cat', 'indoor-outdoor cat', 'indoor outdoor cat',
                     'adopted cat', 'abused cat', 'abyssinian', 'angora cat', 'egyptian mau',
                     'egyptian cat']

cat_filter_list_2 = ['😻', '😸', '😺', '😹', '🙀', '😿', '😾', '😽', '😼', 'meaw', 
                     'purr', 'feline', 'кошка', '고양이', 'gato', 'con mèo', 'ネコ', '猫']

phrases_cat_ownership = 'my '+'|my '.join(cat_filter_list)+\
'|I have a '+'|I have a '.join(cat_filter_list)+\
'|I adopted a '+'|I adopted a '.join(cat_filter_list)+\
'|I rescued a '+'|I rescued a '.join(cat_filter_list)+\
'|I have an '+'|I have an '.join(cat_filter_list_1)+\
'|I have two '+'|I have two '.join(cat_filter_list)+\
'|I got a '+'|I got a '.join(cat_filter_list)+\
'|I own a '+'|I own a '.join(cat_filter_list)+\
'|моя кошка'+'|у меня есть кошка'+'|내 고양이'+'|나는 고양이가있다'+\
'|mi gato'+'|tengo un gato'+'|con mèo của tôi'+'|tôi có một con mèo'+\
'|我的猫'+'|我有一只猫'+'|私の猫'+'|私は猫を飼っている'

cat_filter_list += cat_filter_list_1 + cat_filter_list_2
dog_filter_list += dog_filter_list_1 + dog_filter_list_2

cat_stopwords = phrases_cat_ownership.replace('|', ',').split(",")
dog_stopwords = phrases_dog_ownership.replace('|', ',').split(",")

feature_list = [('comment','count'),('comment','avg_length'),('comment','avg_nemojis'),
                ('comment','avg_npunctuations'),('comment','avg_ntags'),
                ('comment','uses_dog_keyword'),('comment','uses_cat_keyword'),
                ('creator_name','nunique'),('creator_name','contains_dog_keyword'),
                ('creator_name','contains_cat_keyword')]

features_to_scale = [('comment','count'),('comment','avg_length'),
                     ('comment','avg_nemojis'),('comment','avg_npunctuations'),
                     ('comment','avg_ntags'),('creator_name','nunique')]