#include <pthread.h>
#include <stdio.h>

void *new_thread_func(void *arg)
{
  printf("This is thread %ld\n", pthread_self());
  return NULL;
}

int main()
{
  pthread_t thread;
  int result = pthread_create(&thread, NULL, new_thread_func, NULL);
  if (result != 0)
  {
    printf("Thread creation failed\n");
    return 1;
  }
  pthread_join(thread, NULL);
  return 0;
}