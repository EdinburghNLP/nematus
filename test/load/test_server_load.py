#!/usr/bin/env python

import json
import random

from locust import HttpLocust, TaskSet, task

test_segments = []
with open('../data/corpus.en') as f:
    for line in f:
        test_segments.append(line.strip())

def get_random_test_segments(n=1):
    return random.sample(test_segments, n)

class TranslationTasks(TaskSet):
    def on_start(self):
        self.client.get("/status")

    @task
    def translate_10_random_segments(self):
        segments_source = get_random_test_segments(10)
        headers = {'content-type': 'application/json'}
        payload = json.dumps({
            "segments": segments_source,
            "return_word_alignment": False,
            "return_word_probabilities": False,
        })
        self.client.post("/translate", payload, headers=headers)

class TranslationClient(HttpLocust):
    task_set = TranslationTasks
    min_wait = 5000
    max_wait = 15000
