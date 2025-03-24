#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM_THREADS 5

pthread_mutex_t mutex;
int shared_resource = 0;

void *thread_function(void *thread_id)
{
  long tid = (long)thread_id;

  pthread_mutex_lock(&mutex);
  shared_resource++;
  printf("Thread %ld: shared_resource is now %d\n", tid, shared_resource);
  pthread_mutex_unlock(&mutex);

  pthread_exit(NULL);
}

int main()
{
  pthread_t threads[NUM_THREADS];
  int rc;
  long t;

  pthread_mutex_init(&mutex, NULL);

  for (t = 0; t < NUM_THREADS; t++)
  {
    printf("Main: creating thread %ld\n", t);
    rc = pthread_create(&threads[t], NULL, thread_function, (void *)t);
    if (rc)
    {
      printf("Error: unable to create thread, %d\n", rc);
      exit(-1);
    }
  }

  for (t = 0; t < NUM_THREADS; t++)
  {
    pthread_join(threads[t], NULL);
  }

  printf("Main: program completed. shared_resource = %d.\n", shared_resource);

  pthread_mutex_destroy(&mutex);
  pthread_exit(NULL);
}