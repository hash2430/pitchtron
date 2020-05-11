train_list = '/home/admin/projects/mellotron_as_is/filelists/extended_merge_korean_pron_train.txt'
reference_sentence='위반시 벌칙끔 십삼마 눤 십삼세 미만 융마 눠니 부과됨니더 육쎄 미만 카씨트 미차굥시에는 벌칙끔 융마 눠니 부과됨니더'
reference_sentence='팔번 아이언과 구번 아이어는 숃 아이어니고 정왁또가 노픈 클러빔니다 그리네 가까울쑤록 거리와 방양에 조저리 중요해지므로 더 마니 사용암니다'
short_lines_only = []
with open(train_list, 'r', encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    columns = line.split('|')
    if len(columns[1]) < len(reference_sentence):
        short_lines_only.append(line)

new_file_list_name = '/home/admin/projects/mellotron_as_is/filelists/wav_less_than_12s_158_speakers_train.txt'
with open(new_file_list_name, 'w', encoding='utf-8') as f:
    for line in short_lines_only:
        f.write(line)