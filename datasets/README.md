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
