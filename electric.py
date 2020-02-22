#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 15:59:13 2016

@author: mzhong
"""
import numpy as np
import pandas as pd
import datetime

class submeter(object):
        
    def timedelta64_to_secs(self, timedelta):
        """Convert `timedelta` to seconds.
    
        Parameters
        ----------
        timedelta : np.timedelta64
    
        Returns
        -------
        float : seconds
        """
        if timedelta is None:
            return np.array([])
        else:
            return timedelta / np.timedelta64(1, 's')  
            
    def removeDuplicates(self, eventDF):
        # Remove all but first in consecutive series of offs or ons in a dataframe of ons and offs
        # Could occur after removal of events under min length
        eventDF['change_duplicate'] = eventDF['state_change'].diff()
        eventDF = eventDF[ eventDF['change_duplicate'] != 0 ]
        return eventDF
        
    def calculatePeriods(self, eventDF, endtime):
        # Calculate duration of on/off events. (Time of next event minus time of event.)
        eventDF['duration'] = -eventDF['time'].diff(-1)
        # Last event is a special case as there's no time of next event
        # Use end of sample time instead. If the event is very close to the end of the sameple
        # time, it will get discarded as too short when running the rule but will be picked up
        # as part of the overlap sample on the next run.
        lastDuration = endtime - eventDF.iloc[-1, eventDF.columns.get_loc('time')]
        eventDF.iloc[-1, eventDF.columns.get_loc('duration')] = lastDuration
        # Convert to seconds
        eventDF['duration'] = self.timedelta64_to_secs(eventDF['duration'])
        return eventDF
    
    # Take a Series of readings and analyse to find which are on or off events
    # Return DataFrame containing times and event types ('off' or 'on')
    
#     Most appliances spend a lot of their time off.  This function finds
#         periods when the appliance is on.
#     
#         Parameters
#         ----------
#         chunk : pd.Series
#         end : end of sample time period  
#         min_off_duration : int
#             If min_off_duration > 0 then ignore 'off' periods less than
#             min_off_duration seconds of sub-threshold power consumption
#             (e.g. a washing machine might draw no power for a short
#             period while the clothes soak.)  Defaults to 0.
#         min_on_duration : int
#             Any activation lasting less seconds than min_on_duration will be
#             ignored.  Defaults to 0.
#         on_power_threshold : int or float
#             Watts

    def get_ons_and_offs(self, chunk, end, prevEvent, min_off_duration=0, min_on_duration=0,
                            max_on_duration=20000, on_power_threshold=5, dx=False):
        # Convert Series of times and power values to DataFrame for manipulation.
        eventDF = pd.DataFrame({'time':chunk.index, 'value':chunk.values})
        # Determine if appliance off or on for each reading.
        eventDF['when_on'] = (eventDF['value'] >= on_power_threshold)
        
        # Find state changes
        # Get series of 1, -1, 0 - switch on, switch off, no change
        # State change for first reading will be undefined.
        eventDF['state_change'] = eventDF['when_on'].astype(np.int8).diff()
        if (eventDF.empty):
            return eventDF
        
        # Assume first reading is a state change to its current state.
        # Will need to check whether to discard later.
        if (eventDF.iloc[0, eventDF.columns.get_loc('when_on')]):
            eventDF.iloc[0, eventDF.columns.get_loc('state_change')] = 1
        else:
            eventDF.iloc[0, eventDF.columns.get_loc('state_change')] = -1
        
        del eventDF['when_on']
        
        # Discard everything which isn't a switch on or off.  
        eventDF = eventDF[ (eventDF['state_change'] == 1) | (eventDF['state_change'] == -1) ]
        # Smooth over off-durations less than min_off_duration
        if (min_off_duration > 0) | (min_on_duration > 0):
            eventDF = self.calculatePeriods(eventDF, end)
            
            # Remove short offs first, get rid of excess ons, recalculate periods.
            # Ons including short offs are thus counted as longer ons
            if (min_off_duration > 0):
                # If there is an off that just spans the end of the period it will look short
                # even though it isn't. This could lead to a false on. (Short on, followed by
                # a few seconds of off. Off is removed and so the on becomes longer.) If there's
                # a short off right at the end, move end time of period earlier to compensate.
                if (eventDF.iloc[-1, eventDF.columns.get_loc('time')] > (end - datetime.timedelta(seconds=min_off_duration))) &\
                                (eventDF.iloc[-1, eventDF.columns.get_loc('duration')] < min_off_duration) &\
                                             (eventDF.iloc[-1, eventDF.columns.get_loc('state_change')] == -1):
                     end = eventDF.iloc[-1, eventDF.columns.get_loc('time')]
                                
                # Remove short offs
                eventDF = eventDF [ ~( (eventDF['state_change'] == -1) & (eventDF['duration'] < min_off_duration) )]
                eventDF = self.removeDuplicates(eventDF)
                # Recalculate periods now some events may have been removed.
                eventDF = self.calculatePeriods(eventDF, end)
            if (max_on_duration):
                longOns = pd.DataFrame(columns=eventDF.columns)
                onsList = []
                # Find ons that are longer than the maximum on
                for indexOnOff, onOff in eventDF.iterrows():
                    if (onOff['state_change'] == 1) & (onOff['duration'] > max_on_duration):
                        onsList.append(onOff.values)
                longOns = longOns.append(pd.DataFrame(onsList, columns=eventDF.columns)).reset_index()
                # For each of the long ons try to find a different on.
                for indexLongOn, longOn in longOns.iterrows():
                    retryEnd = longOn['time'] + datetime.timedelta(seconds=(longOn['duration']))
                    retryDF = pd.DataFrame({'time':chunk.index, 'value':chunk.values})
                    # Get the readings for the period of the long on
                    retryDF = retryDF[ (retryDF['time'] >= longOn['time']) & (retryDF['time'] <= retryEnd) ]
                    if not retryDF.empty:
                        retryDF = retryDF.reset_index(drop=True)
                        # Calculate period of each reading.
                        retryDF = self.calculatePeriods(retryDF, retryEnd)
                        newOnIndex = None
                        rows = 0
                        # Find last reading with a period more than max
                        for indexReadRow, readRow in retryDF.iterrows():
                            if (readRow['duration'] > max_on_duration):
                                newOnIndex = indexReadRow + 1
                            rows = rows + 1
                        # Find the next on after that. Will be the next reading as all ons
                        # unless it's the final off (in which case move on).
                        if (newOnIndex is not None):
                            if (newOnIndex in retryDF.index):
                                if not ((newOnIndex == rows - 1) & (retryDF.iloc[newOnIndex, eventDF.columns.get_loc('state_change')] == 0)):
                                    eventDF = eventDF.append(retryDF.iloc[[newOnIndex]], ignore_index=True)
                                    eventDF.iloc[-1, eventDF.columns.get_loc('state_change')] = 1
                if not longOns.empty:
                    eventDF = pd.concat([eventDF, longOns])
                    del eventDF['index']
                    eventDF.drop_duplicates(keep=False, inplace=True)
                    if not eventDF.empty:
                        eventDF = eventDF.sort_values(['time'])
                        eventDF = self.calculatePeriods(eventDF, end)
            #Remove ons which are still short
            if (min_on_duration > 0):
                #Remove short ons
                eventDF = eventDF [ ~( (eventDF['state_change'] == 1) & (eventDF['duration'] < min_on_duration) )]
                eventDF = self.removeDuplicates(eventDF)
                #Don't need to bother recalculating period as not used again
        del eventDF['duration']
        del eventDF['value']
        del eventDF['change_duplicate']
        eventDF['state_change'] = eventDF['state_change'].map({1:'on',-1:'off'})
        eventDF = eventDF.reset_index()            
        # Potentially drop first event if it matches the last one previously recorded.
        if (not eventDF.empty) & (prevEvent is not None):
            if prevEvent.eventtype == eventDF.iloc[0, eventDF.columns.get_loc('state_change')]:
                # Repeat off is no use. Drop it.
                if prevEvent.eventtype == 'off':
                    eventDF.drop(eventDF.index[:1], inplace=True)
                else:
                    # Drop repeat on if time since last one is less than max on duration.
                    if (eventDF.iloc[0, eventDF.columns.get_loc('time')] - prevEvent.time) < datetime.timedelta(seconds=max_on_duration):
                        eventDF.drop(eventDF.index[:1], inplace=True)
        #Columns time and state_change (either 'on' or 'off')        
        return eventDF
            
    def energy(self, window, power_unit=0.1):
            # compute the total energy used
            # input: window is a pandas Series. Index: time, Value: power
            # output: energy in watt-hours
            
            # compute the duration of each time step
            try:
                # Get length of time between each reading.
                time_diff = np.float64(pd.DataFrame(window.index).diff()[0:]
                                   .astype('timedelta64[s]').values.
                                   flatten()/3600.0)[1:]          
                # Adjust power value for unit type,
                power = power_unit*window.values[0:-1].flatten()
                # Energy used is power * time.
                energy_used = np.dot(time_diff,power)
                return energy_used
            except IndexError:
                return 0

    # Use a set of onOff events and a set of power readings covering
    # period from at least the time of the first on event until the time
    # of last off event to create BAMs.
    def get_bams_from_on_offs(self,onOffs,rawReadings,power_unit=0.1):
        columns = ['start time',
                   'end time',
                   'duration (seconds)',
                   'energy (Whs)']
        df_bam = pd.DataFrame(columns=columns)
        # Loop through off events.
        for i in range(1, onOffs.shape[0], 2):
            offEventTime = onOffs.iloc[i, onOffs.columns.get_loc('time')]
            onEventTime = onOffs.iloc[i-1, onOffs.columns.get_loc('time')]
            # Get all readings for appliance between on and off times, inclusive.
            windowReadings = rawReadings.loc[onEventTime:offEventTime]
            array = [[str(onEventTime),
                     str(offEventTime),
                     self.timedelta64_to_secs(offEventTime - onEventTime),
                     self.energy(windowReadings,power_unit=power_unit)]]
            # Add new BAM to the list.
            new_df = pd.DataFrame(array,columns=columns)
            df_bam = new_df.append(df_bam, ignore_index=True)
        df_bam = df_bam.sort_values(['start time'])
        return df_bam
    