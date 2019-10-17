
data=["I am Mohammed Abacha,the son of the late Nigerian Head of "
      "State who died on the 8th of June 1998. Since i have been "
      "unsuccessful in locating the relatives for over 2 years now"
      " I seek your consent to present you as the next of kin so "
      "that the proceeds of this account valued at US$15.5 Million"
      " Dollars can be paid to you. If you are capable and willing"
      " to assist, contact me at once via email with following "
      "details: 1. Your full name, address, and telephone number."
      " 2. Your Bank Name, Address. 3. Your Bank Account Number and "
      "Beneficiary Name-You must be the signatory."]

#首先需要将邮件量化
from sklearn.feature_extraction.text import CountVectorizer
vec=CountVectorizer()
X=vec.fit_transform(data)

#结果的出现次数多寡的单词是根据首字母进行排序的;
out1=vec.get_feature_names()
# print(X)
# print(out1)

out2=X.toarray()
# print(out2)

# 只要数据的前5即可;
out3=X.toarray()[0,:5]
# print(out3)

