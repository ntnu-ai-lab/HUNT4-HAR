import dispy, random
import time, socket


# 'compute' is distributed to each node running 'dispynode'
def compute(n):
    time.sleep(n)
    host = socket.gethostname()
    return (host, n)

if __name__ == '__main__':
    cluster = dispy.JobCluster(compute)
    jobs = []
    for i in range(4):
        # schedule execution of 'compute' on a node (running 'dispynode')
        # with a parameter (random number in this case)
        job = cluster.submit(random.randint(5,7))
        job.id = i # optionally associate an ID to job (if needed later)
        jobs.append(job)
    # cluster.wait() # waits for all scheduled jobs to finish
    for job in jobs:
        host, n = job() # waits for job to finish and returns results
        print('%s executed job %s at %s with %s' % (host, job.id, job.start_time, n))
        # other fields of 'job' that may be useful:
        # print(job.stdout, job.stderr, job.exception, job.ip_addr, job.start_time, job.end_time)
    cluster.print_status()