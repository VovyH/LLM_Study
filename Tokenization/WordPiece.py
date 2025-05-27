class WordpieceTokenizer:
    def tokenize(self, text):
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                # 太长了就直接标记为 [UNK]
                output_tokens.append(self.unk_token)
                continue
            is_bad = False
            start = 0
            sub_tokens = []
            # 循环，尽量从当前位置往后找最大匹配
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr  # 词首后面的子词加 "##"
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                # 如果没找到任何子串匹配，则该 token 整体标记为 [UNK]
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end
            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens