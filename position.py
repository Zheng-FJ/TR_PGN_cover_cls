import re  

def position_collection(key:str, value:str):
    """ Return the position of key 
    in the value.
    """
    def position_finder(k_):
        return [i.start() for i in re.finditer(k_, value)]
    
    def multi_position(ref, selection):
        if type(ref) == str:
            return selection[0]
        for item in selection:
            if ref <= item:
                return item
        return selection[0]
        
    key_, idx, collection = ("", 0, [])
    GENERATED_TOKEN = "Generated"
    
    current_p = 0
    while idx < len(key):
        key_ += key[idx]
        position_ =  position_finder(key_)
        idx += 1
        
        if len(position_) == 0 and len(key_) == 1:
            collection.append([key_, GENERATED_TOKEN])
            key_ = ""
        elif len(position_) == 0 and len(key_) > 1:
            key_ = key_[:-1]
            position_ = position_finder(key_)
            # ref_num = collection[-1][-1] if collection else 0
            p_ = multi_position(ref=current_p, selection=position_)
            len_key = len(key_)
            collection.append([key_, p_, len_key])
            current_p = p_
            idx -= 1
            key_ = ""
        elif len(position_) > 0 and key_[-1] == "。":
            p_ = multi_position(ref=current_p, selection=position_)
            len_key = len(key_)
            collection.append([key_, p_, len_key])
            current_p = p_
            key_ = ""
        
    else:
        if len(key_) == 0:
            pass
        elif len(key_) == 1 and key_ not in value:
            collection.append([key_, GENERATED_TOKEN])
        else:
            # ref_num = collection[-1][-1] if collection else 0
            p_ = multi_position(ref=current_p, selection=position_)
            len_key = len(key_)
            collection.append([key_, p_, len_key])
    return collection

if __name__ == "__main__":

    summarized_text = "警察はこのよう。各社の成長を支えているのが非通信事業の伸びである。NTTドコモの非通信事業が売上に占める割合は、2019年度上半期は20.2％と前年度の18.4％から1.8％増、KDDIは22.8％と同5.7％増、ソフトバンクは22.7％と同0.1％増を記録した。"
    original_text = "携帯キャリアの2019年度上半期の決算は、NTTドコモが減収減益、KDDIが増収減益、ソフトバンクが増収増益と明暗が分かれた。 　10月からは事業法が改正されたが、これを巧みに集客へ結びつけることに成功させたところがある一方で、いち早く改正に対応したことで割りを喰ったところもあったようだ。 　市場のルールや競争とは何なのか。なぜ、こうも取り組みに違いが生まれたのか。あらためて考えさせられたイベントではなかっただろうか。 　今回は携帯キャリアの売上から、通信事業と非通信事業に分類し、非通信事業の成長性について取り上げてみたい。 　事業法改正などの影響もあり、携帯キャリアの通信事業は年々厳しさを増している。前年同期と比較してNTTドコモとKDDIは売上、営業利益ともマイナス傾向となる一方で、ソフトバンクはプラスを維持。その一方で、各社の成長を支えているのが非通信事業の伸びである。 　NTTドコモの非通信事業が売上に占める割合は、2019年度上半期は20.2％と前年度の18.4％から1.8％増、KDDIは22.8％と同5.7％増、ソフトバンクは22.7％と同0.1％増を記録した。 　このように、各社の非通信事業の売上は20％程度まで拡大しているが、2020年度はヤフーとLINEが経営統合することで、ソフトバンクの非通信事業は頭一つ抜けることが確実視される。 　8000万人の利用者を持つLINEがソフトバンク傘下に入ったことで、決済やEC、経済圏を巡る非通信事業の戦いは楽天も含め、来年以降更に激化していくこととなりそうだ。"
    res = position_collection(summarized_text, original_text)
    print(res)
