# Input
본문 문단 별 텍스트 및 질문
```json
{'version': 'v2.0',
 'data':[{
        'content_id'                                    // 본문 (여러 개의 문단을 포함할 수 있음) id
        'title'                                                  // 본문의 제목
        'paragraphs':[{                              // 본문 내 문단 리스트
                'paragraph_id'                      // 각 문단의 id
                'context'                                  // 각 문단의 내용
                'qas':[                                        // 각 문단으로 응답해야 할 질문-답변 리스트
                        {'question_id'                // 각 문단의 id
                         'question'                       // 질문 query
                         'answers'[{                     // 각 질문의 답변 리스트
                                 'text'                         // 응답 문자열
                                 'answer_start'}]  // 문단에서 응답이 위치하는 시작 인덱스
                         'is_impossible'}]         // 질문이 해당 문단으로 응답될 수 있는지 여부
                        }]
                }]
        }]
}
```
Output
각 질문의 답 (텍스트)

Submission 양식
```
question_id, answer_text
QUES_xxxxxxxxxx, ans
QUES_xxxxxxxxxx, ans
QUES_xxxxxxxxxx, ans
QUES_xxxxxxxxxx, ans
QUES_xxxxxxxxxx, ans
...
```
✔ sample_submission과 같은 형태의 csv(utf8) 파일 1개를 제출 (sample_submission.csv는 대회 시작 때 제공)

✔ 'question_id', 'answer_text' 두 가지 column이 필요함

- 'question_id' : 질문의 고유번호

- 'answer_text' : 질문에 대한 답 문자열

✔ 제출 파일의 question_id 순서가 sample submission.csv와 동일해야 함

✔ 제출 파일에 누락된 question_id가 없어야 함

✔ 정답 텍스트에서 띄어쓰기는 고려하지 않고 채점함
