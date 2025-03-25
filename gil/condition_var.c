
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define BUFFER_SIZE 5

int buffer[BUFFER_SIZE];
int count = 0;

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond_producer = PTHREAD_COND_INITIALIZER;
pthread_cond_t cond_consumer = PTHREAD_COND_INITIALIZER;

void *producer(void *arg)
{
  int item;
  while (1)
  {
    item = rand() % 100;
    pthread_mutex_lock(&mutex);

    while (count == BUFFER_SIZE)
    {
      pthread_cond_wait(&cond_producer, &mutex);
    }

    buffer[count] = item;
    count++;
    printf("Producer produced %d, buffer count: %d\n", item, count);

    pthread_cond_signal(&cond_consumer);
    pthread_mutex_unlock(&mutex);

    sleep(rand() % 3);
  }
  return NULL;
}

void *consumer(void *arg)
{
  int item;
  while (1)
  {
    pthread_mutex_lock(&mutex);

    while (count == 0)
    {
      pthread_cond_wait(&cond_consumer, &mutex);
    }

    item = buffer[count - 1];
    count--;
    printf("Consumer consumed %d, buffer count: %d\n", item, count);

    pthread_cond_signal(&cond_producer);
    pthread_mutex_unlock(&mutex);

    sleep(rand() % 3);
  }
  return NULL;
}

int main()
{
  pthread_t producer_thread, consumer_thread;

  if (pthread_create(&producer_thread, NULL, producer, NULL) != 0)
  {
    perror("Failed to create producer thread");
    return 1;
  }

  if (pthread_create(&consumer_thread, NULL, consumer, NULL) != 0)
  {
    perror("Failed to create consumer thread");
    return 1;
  }

  pthread_join(producer_thread, NULL);
  pthread_join(consumer_thread, NULL);

  // unreachable code
  return 0;
}