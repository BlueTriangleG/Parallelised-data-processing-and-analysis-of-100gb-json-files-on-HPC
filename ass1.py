from collections import defaultdict
from decimal import Decimal
import time
import simplejson as json
from mpi4py import MPI
import os
# read the raws from the file and calculate the sum of sentiment scores for each hour

def find_sum_hour(filename, size, rank):
    hourly_sum = defaultdict(int)
    fullname = f'{filename}.json'
    # Get the size of the file and calculate the average size of each part
    file_size = os.path.getsize(fullname)
    avg_size = file_size // size
    skip_size = avg_size * rank
    print(f"The size of the file is {file_size} bytes.")
    with open(fullname, 'rb') as f:
        f.seek(skip_size)  # Move to the approximate position
        f.readline()
        while True:
            position = f.tell()
            # if the position is greater than the end of the part, break
            if position >= skip_size + avg_size:
                break
            line = f.readline().strip()
            # convert bytes to string, refomate the string
            line = line.decode('utf-8')
            line = line[:-1]
            try:
                # convert JSON string to Python dictionary
                tweet = json.loads(line)
                doc_data = tweet.get('doc', {}).get('data', {})
                sentiment = doc_data.get('sentiment', 0)
                if not isinstance(sentiment, (int, float, Decimal)):
                    sentiment = Decimal(0)
                else:
                    sentiment = Decimal(str(sentiment))  # convert all numbers to Decimal
                created_at = doc_data.get('created_at')
                if created_at:
                    hour = created_at.split(':')[0]  # Extract hour part from timestamp
                if hour not in hourly_sum:
                    hourly_sum[hour] = [0, 0]  # Initialize hourly sum if not exists
                hourly_sum[hour][0] += 1
                hourly_sum[hour][1] += sentiment
            except:
                print("Reach the end")
                break

    return hourly_sum

def find_happiest_and_most_active(hourly_sum):
    # Find the happiest hour based on sum of sentiment
    happiest_hour, happiest_hour_sum = max(hourly_sum.items(), key=lambda x: x[1][1])
    daily_sum = defaultdict(int)
    for hour, sentiment in hourly_sum.items():
        day = hour.split('T')[0]
        daily_sum[day] += sentiment[1]
    happiest_day, happiest_day_sum = max(daily_sum.items(), key=lambda x: x[1])
    most_active_hour, most_active_hour_tweet_count = max(hourly_sum.items(), key=lambda x: x[1][0])
    daily_tweet_count = defaultdict(int)
    for hour, sentiment in hourly_sum.items():
        day = hour.split('T')[0]
        daily_tweet_count[day] += sentiment[0]
    most_active_day, most_active_day_tweet_count = max(daily_tweet_count.items(), key=lambda x: x[1])
    return happiest_hour, happiest_hour_sum, happiest_day, happiest_day_sum, most_active_hour, most_active_hour_tweet_count, most_active_day, most_active_day_tweet_count

def merge_hourly_sums(hourly_sums):
    # Merge hourly sums from all ranks
    merged_hourly_sum = defaultdict(lambda: [0, Decimal('0')])
    for hourly_sum in hourly_sums:
        for hour, values in hourly_sum.items():
            merged_hourly_sum[hour][0] += values[0]
            merged_hourly_sum[hour][1] += values[1]
    return merged_hourly_sum

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    filename = 'twitter-50mb'
    # every rank reads its own part of the file and calculates the sum of sentiment scores for each hour
    hourly_sum = find_sum_hour(filename, size, rank)
    comm.Barrier()
    if rank == 0:
        hourly_sums = [hourly_sum]
        for i in range(1, size):
            hourly_sums.append(comm.recv(source=i, tag=i))
        merged_hourly_sum = merge_hourly_sums(hourly_sums)
        happiest_hour, happiest_hour_sum, happiest_day, happiest_day_sum, most_active_hour, \
            most_active_hour_tweet_count, most_active_day, most_active_day_tweet_count = \
                find_happiest_and_most_active(merged_hourly_sum)
        print("Happiest hour:", happiest_hour)
        print("Happiest hour sentiment sum:", happiest_hour_sum[1])
        print("Happiest day:", happiest_day)
        print("Happiest day sentiment sum:", happiest_day_sum)
        print("Most active hour:", most_active_hour)
        print("Number of tweets in the most active hour:", most_active_hour_tweet_count[0])
        print("Most active day:", most_active_day)
        print("Number of tweets in the most active day:", most_active_day_tweet_count)
        end_time = time.time()
        print("Total runtime: {:.2f} seconds".format(end_time - start_time))
    else:
        comm.send(hourly_sum, dest=0, tag=rank)

if __name__ == '__main__':
    start_time = time.time()
    main()