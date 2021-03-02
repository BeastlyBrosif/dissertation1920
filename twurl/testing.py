import json
import subprocess as commands

import time

def get_followers(screen_name):

    followers_list = []

    # start cursor at -1
    next_cursor = -1

    print("Getting list of followers for user '%s' from Twitter API..." % screen_name)

    while next_cursor:

        cmd = 'twurl "/1.1/followers/ids.json?cursor=' + str(next_cursor) + \
                '&screen_name=' + screen_name + '"'
        cmd = 'twurl "/1.1/statuses/user_timeline.json?screen_name='+screen_name+'&exclude_replies=true&include_rts=false"'
        (status, output) = commands.getstatusoutput(cmd)
        # convert json object to dictionary and ensure there are no errors
        try:
            data = json.loads(output)

            if data.get("errors"):

                # if we get an inactive account, write error message
                if data.get('errors')[0]['message'] in ("Sorry, that page does not exist",
                                                        "User has been suspended"):

                    print("Skipping account %s. It doesn't seem to exist" % screen_name)
                    break

                elif data.get('errors')[0]['message'] == "Rate limit exceeded":
                    print("\t*** Rate limit exceeded ... waiting 2 minutes ***")
                    time.sleep(120)
                    continue

                # otherwise, raise an exception with the error
                else:

                    raise Exception("The Twitter call returned errors: %s"
                                    % data.get('errors')[0]['message'])

            if data.get('ids'):
                print("\t\tFound %s followers for user '%s'" % (len(data['ids']), screen_name))
                followers_list += data['ids']

            if data.get('next_cursor'):
                next_cursor = data['next_cursor']
            else:
                break

        except ValueError:
            print("\t****No output - Retrying \t\t%s ****" % output)

    return followers_list


num_of_users = 0;
with open("/user_list.txt", 'a', encoding="utf-8") as file:
    for line in file:
        num_of_users = num_of_users + 1
    file.close()